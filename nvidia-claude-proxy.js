const http = require('http');
const fs = require('fs');
const path = require('path');

const CONFIG_PATH = process.env.PROXY_CONFIG ? path.resolve(process.env.PROXY_CONFIG) : path.join(__dirname, 'nvidia-proxy-config.json');
let cfg = {};
try { cfg = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')); } catch (e) { cfg = {} }

const NVIDIA_BASE = (process.env.NVIDIA_BASE_URL || cfg.nvidiaBaseUrl || 'https://integrate.api.nvidia.com/v1').replace(/\/+$/, '');
const NVIDIA_KEY = process.env.NVIDIA_API_KEY || cfg.nvidiaApiKey || '';
const FORCE_MODEL = process.env.FORCE_MODEL || cfg.model || 'deepseek-ai/deepseek-v4-flash';
const HOST = process.env.HOST || cfg.listenHost || '127.0.0.1';
const PORT = parseInt(process.env.PORT || cfg.listenPort || '3456', 10);
const UPSTREAM_TIMEOUT = parseInt(process.env.UPSTREAM_TIMEOUT || cfg.upstreamTimeout || '120000', 10);
// Local backend (e.g. Ollama / LM Studio) — OpenAI-compatible, no real auth needed.
const LOCAL_BASE_URL = (process.env.LOCAL_BASE_URL || cfg.localBaseUrl || 'http://127.0.0.1:11434/v1').replace(/\/+$/, '');
const LOCAL_API_KEY = process.env.LOCAL_API_KEY || cfg.localApiKey || 'ollama';
const LOCAL_MODEL = process.env.LOCAL_MODEL || cfg.localModel || 'llama-8b';
const LOCAL_FALLBACK = /^true$/i.test(String(process.env.LOCAL_FALLBACK_TO_CLOUD || cfg.localFallbackToCloud || 'false'));
// Local context window baked into Modelfile.qwythos (num_ctx). On an 8GB RTX 4060 the
// qwythos-9b Q4 weights (~6.5GB) leave almost no VRAM for KV cache, so Ollama offloads the
// KV to CPU — functional, but this is the hard ceiling the proxy must fit prompts under.
const LOCAL_NUM_CTX = parseInt(process.env.LOCAL_NUM_CTX || cfg.localNumCtx || '16384', 10) || 16384;
// Cap on how many tokens the LOCAL model generates per call. Claude Code ships a large
// max_tokens (often the model's full window, e.g. 32000), but a 9B local model shouldn't
// emit that much — it's slow and the proxy translation doesn't need it. We cap the local
// reply and, importantly, derive the prompt limit from THIS cap rather than the raw
// body.max_tokens, otherwise the limit collapses to ~7k and even system+tools won't fit.
const LOCAL_REPLY_CAP = parseInt(process.env.LOCAL_REPLY_CAP || cfg.localReplyCap || '4096', 10) || 4096;
// OPTIONAL hard ceiling on the total estimated prompt (conversation + tool + template
// overhead). 0 / unset => derive automatically from LOCAL_NUM_CTX (recommended). Set it
// lower only if you want the proxy to drop context more aggressively.
const LOCAL_MAX_PROMPT = parseInt(process.env.LOCAL_MAX_PROMPT_TOKENS || cfg.localMaxPromptTokens || '0', 10) || 0;
// Recursive summary-based compaction for local models (the "auto /compact" behavior the
// user asked for): instead of just DROPPING oldest turns, the proxy asks the local model to
// SUMMARIZE them into a running digest, and loops until the prompt fits. Preserves context
// the drop-only strategy would forget. Disable with LOCAL_SUMMARIZE=false if the extra
// latency (one local generation per compaction round) is unacceptable.
// NOTE: cfg[key] may legitimately be the boolean `false`; reading it through `|| default`
// would treat that falsy value as "unset" and wrongly fall back to the default. So we test
// `!== undefined` explicitly before applying the truthy/falsy default.
function boolCfg(envName, cfgKey, dflt) {
  if (process.env[envName] !== undefined) return /^true$/i.test(process.env[envName]);
  if (cfg[cfgKey] !== undefined) return /^true$/i.test(String(cfg[cfgKey]));
  return dflt;
}
const LOCAL_SUMMARIZE = boolCfg('LOCAL_SUMMARIZE', 'localSummarize', true);
const LOCAL_SUMMARIZE_ROUNDS = parseInt(process.env.LOCAL_SUMMARIZE_ROUNDS || cfg.localSummarizeRounds || '10', 10) || 10;
// Optional dedicated summarizer model (must exist in the local backend). If empty, the main
// LOCAL_MODEL is used for summarization too. A non-reasoning model (e.g.
// mannix/llama3.1-8b-abliterated) yields cleaner digests because its text lands in `content`
// rather than being leaked into `reasoning`, but it is optional.
const LOCAL_SUMMARIZER = (process.env.LOCAL_SUMMARIZER || cfg.localSummarizerModel || '').trim() || '';
// Claude Code ships its FULL tool set (often 80+ tools) on every request. Rendered through
// the local model's chat template those tool schemas alone can cost 30k+ tokens — bigger than
// the whole 16384 window even with zero conversation. For local routes we cap each tool's (and
// each schema field's) `description` so the fixed tool overhead shrinks enough to fit. Tool
// NAMES and parameter STRUCTURE are preserved, so the model can still emit valid tool calls.
const LOCAL_TOOL_TRIM = boolCfg('LOCAL_TOOL_TRIM', 'localToolTrim', true);
const LOCAL_TOOL_DESC_MAX = parseInt(process.env.LOCAL_TOOL_DESC_MAX || cfg.localToolDescMax || '60', 10) || 60;
// For local routes, optionally DROP the tool definitions entirely (pure-chat local session).
// A 9B model can't usefully wield Claude Code's 80+ tools, and their schemas alone cost ~32k
// tokens — bigger than the whole 16384 window. Dropping them makes the local session fast and
// always fit. Set false (and raise num_ctx + recreate the Ollama model) to keep tool calling.
const LOCAL_DROP_TOOLS = boolCfg('LOCAL_DROP_TOOLS', 'localDropTools', true);
// Sentinel returned by callUpstream when a LOCAL call exceeded the context window and the
// caller should re-compact and retry (instead of the proxy committing a 400 to the client).
const LOCAL_CONTEXT_EXCEEDED = Symbol('local-context-exceeded');

// Fallback list (confirmed-valid ids) used until the live upstream catalog is fetched.
// Tencent MaaS ids are listed first so they survive even if the catalog fetch fails.
let MODELS = [
  { id: 'glm-5.2', display_name: 'GLM 5.2 (reasoning)' },
  { id: 'deepseek-v4-pro', display_name: 'DeepSeek V4 Pro (reasoning)' },
  { id: 'kimi-k3', display_name: 'Kimi K3 (reasoning)' },
  { id: 'deepseek-ai/deepseek-v4-flash', display_name: 'DeepSeek V4 Flash (fast, NVIDIA)' },
  { id: 'z-ai/glm-5.2', display_name: 'GLM 5.2 (z-ai, NVIDIA)' },
  { id: 'nvidia/llama-3.3-nemotron-super-49b-v1', display_name: 'Nemotron Super 49B (balanced)' },
  { id: 'nvidia/llama-3.1-nemotron-ultra-253b-v1', display_name: 'Nemotron Ultra 253B (reasoning)' },
  { id: 'nvidia/nvidia-nemotron-nano-9b-v2', display_name: 'Nemotron Nano 9B (fast)' },
  { id: 'meta/llama-3.3-70b-instruct', display_name: 'Llama 3.3 70B (general)' },
];

// Map Claude Code tier names to real models. haiku (lightweight/aux) -> local
// 8B by default; sonnet/opus (heavy reasoning) -> NVIDIA cloud. Override per
// session via /model (e.g. set Haiku slot to a NVIDIA id) or per message with
// "@local" / "@nvidia".
const TIER_MODELS = {
  haiku: 'local/' + LOCAL_MODEL,
  sonnet: 'z-ai/glm-5.2',
  opus: 'deepseek-ai/deepseek-v4-pro',
};
// Curated fallback pool used when the primary model is exhausted (503/429/timeout).
// deepseek-v4-flash is listed first as the reliable fallback for any primary.
// Override with FALLBACK_MODELS env (comma-separated) for non-NVIDIA upstreams.
const FALLBACK_POOL = (process.env.FALLBACK_MODELS
  ? process.env.FALLBACK_MODELS.split(',').map(s => s.trim()).filter(Boolean)
  : ['deepseek-ai/deepseek-v4-flash', 'z-ai/glm-5.2', 'deepseek-ai/deepseek-v4-pro']);

// Fetch the live NVIDIA model catalog at startup so the switcher reflects what
// build.nvidia.com actually serves. Replaces MODELS with chat-capable models.
const CATALOG_DENY = ['embed', 'guard', 'safety', 'detector', 'translate', 'clip',
  'retriever', 'riva', 'cosmos', 'vila', 'neva', 'parse', 'calibration', 'synthetic',
  'reward', 'kv-cache', 'nv-embed', 'diffusion', 'vision', 'vl', 'omni', 'kosmos', 'fuyu', 'deplot', 'bge', 'arctic'];
async function fetchCatalog() {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 15000);
    const r = await fetch(NVIDIA_BASE + '/models', { headers: { 'Authorization': 'Bearer ' + NVIDIA_KEY }, signal: ctrl.signal });
    clearTimeout(timer);
    if (!r.ok) throw new Error('status ' + r.status);
    const j = await r.json();
    const ids = (j.data || []).map(m => m.id).filter(Boolean);
    const filtered = ids.filter(id => !CATALOG_DENY.some(d => id.toLowerCase().includes(d)));
    if (!filtered.length) throw new Error('empty catalog');
    MODELS = filtered.map(id => ({ id, display_name: id }));
    log('catalog: loaded ' + filtered.length + ' chat-capable NVIDIA models');
  } catch (e) {
    log('catalog fetch failed (' + e.message + '); keeping built-in fallback (' + MODELS.length + ' models)');
  }
}

// Resolve the model id Claude Code sends into a real NVIDIA model id.
function resolveModel(requested) {
  if (!requested) return FORCE_MODEL;
  if (requested.includes('/')) return requested; // provider-prefixed NVIDIA id -> pass through
  if (/^claude-/i.test(requested)) return FORCE_MODEL;
  if (TIER_MODELS[requested]) return TIER_MODELS[requested]; // haiku/sonnet/opus tiers
  // pass through any model present in the live upstream catalog (e.g. Tencent ids like glm-5.2, hy3)
  if (MODELS.some(m => m.id === requested)) return requested;
  return FORCE_MODEL;
}

// Build an ordered fallback chain: requested model first, then distinct backups.
function fallbackChain(primary) {
  const chain = [primary];
  for (const m of FALLBACK_POOL) {
    if (m !== primary && !chain.includes(m)) chain.push(m);
  }
  if (!chain.includes(FORCE_MODEL)) chain.push(FORCE_MODEL);
  return chain;
}

// Scan the most recent user turn for an explicit backend override token
// ("@local" / "@nvidia" / "@本地"). If present, strip it and return the backend.
// This lets the user force a single request to a specific backend.
function detectOverride(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== 'user' || typeof m.content !== 'string') continue;
    const re = /(^|\s)@(local|nvidia|本地)\b/i;
    const mm = m.content.match(re);
    if (mm) {
      m.content = m.content.replace(re, '$1').replace(/\s{2,}/g, ' ').trim() || m.content;
      const key = (mm[2] || '').toLowerCase();
      return (key === 'local' || key === '本地') ? 'local' : 'nvidia';
    }
    break; // only inspect the most recent user turn
  }
  return null;
}

function log(...a) {
  const s = '[' + new Date().toISOString() + '] ' + a.join(' ');
  console.log(s);
  try { fs.appendFileSync(path.join(__dirname, 'nvidia-proxy.log'), s + '\n'); } catch (e) {}
}

function logUsage(model, usage, ms, status) {
  try {
    const u = usage || {};
    fs.appendFileSync(path.join(__dirname, 'usage.jsonl'),
      JSON.stringify({ ts: new Date().toISOString(), model, prompt_tokens: u.prompt_tokens || 0, completion_tokens: u.completion_tokens || 0, ms, status }) + '\n');
  } catch (e) {}
}

// Extract the plain text of a message's content, whether it is a plain string or an
// Anthropic-style array of content blocks (text / tool_use / tool_result). Used by the
// CJK-aware token estimator so nested tool inputs/results are counted too.
function contentText(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    let s = '';
    for (const b of content) {
      if (!b) continue;
      if (typeof b === 'string') s += b;
      else if (typeof b.text === 'string') s += b.text;
      else if (typeof b.content === 'string') s += b.content; // tool_result string body
      else if (b.input != null) s += JSON.stringify(b.input); // tool_use arguments
      else if (Array.isArray(b.content)) s += contentText(b.content); // nested tool_result blocks
    }
    return s;
  }
  return '';
}

// CJK-aware token estimator. Grounded in a real measurement on qwythos-9b:
// 29400 Chinese chars -> 14747 prompt tokens (~2 chars/token). We deliberately
// OVER-estimate (Chinese at 0.6 tok/char vs real ~0.5; Latin at 0.25 vs real ~0.22)
// so the proxy compacts slightly early rather than risking a 400 context overflow.
function estimateTextTokens(text) {
  if (typeof text !== 'string') return 0;
  let cjk = 0, other = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if ((c >= 0x2E80 && c <= 0x9FFF) || (c >= 0x3000 && c <= 0x303F) ||
        (c >= 0xFF00 && c <= 0xFFEF) || (c >= 0x3040 && c <= 0x30FF) ||
        (c >= 0xAC00 && c <= 0xD7AF)) cjk++;
    else other++;
  }
  return Math.ceil(cjk * 0.6 + other * 0.25);
}

function estimateTokens(messages) {
  let n = 0;
  for (const m of messages) {
    n += estimateTextTokens(contentText(m.content)) + 8; // +8: fixed per-message overhead
  }
  return n;
}

// Proxy-level auto-compact for local models: keep the system prompt(s) + most recent
// turns, drop oldest conversation turns until the estimated prompt (including the fixed
// tool-definition + chat-template overhead) fits within `limit`. Stops at the minimal set
// (system + last turn) so it can never loop forever on an unavoidably-huge prompt.
function compactLocalMessages(messages, limit, overhead) {
  const base = overhead || 0;
  const fits = (msgs) => estimateTokens(msgs) + base <= limit;
  if (fits(messages)) return { messages, dropped: 0 };
  const sys = [];
  const rest = [];
  for (const m of messages) {
    if (sys.length === 0 && m.role === 'system') sys.push(m);
    else rest.push(m);
  }
  let dropped = 0;
  while (!fits(sys.concat(rest)) && rest.length > 1) {
    rest.shift();
    dropped++;
  }
  return { messages: sys.concat(rest), dropped };
}

// Minimal survivable prompt: system prompt(s) + the last user/assistant turn (plus the
// turn immediately before it for coherence). This is the last resort when even the first
// compact still overflows the local window — it cannot be reduced further.
function minimalLocalMessages(messages) {
  const sys = messages.filter(m => m.role === 'system');
  let lastIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role !== 'system') { lastIdx = i; break; }
  }
  if (lastIdx < 0) return messages.slice();
  const keep = [];
  if (lastIdx - 1 >= 0 && messages[lastIdx - 1].role !== 'system') keep.push(messages[lastIdx - 1]);
  keep.push(messages[lastIdx]);
  return sys.concat(keep);
}

// Recursively cap every `description` field in a JSON schema to `max` chars. Used to shrink
// tool-definition overhead for local models without altering the schema's structure/validity.
function capSchemaDescriptions(node, max) {
  if (Array.isArray(node)) return node.map(n => capSchemaDescriptions(n, max));
  if (node && typeof node === 'object') {
    const out = {};
    for (const k of Object.keys(node)) {
      const v = node[k];
      if (k === 'description' && typeof v === 'string' && v.length > max) out[k] = v.slice(0, max) + '…';
      else out[k] = capSchemaDescriptions(v, max);
    }
    return out;
  }
  return node;
}

// Build a local-friendly copy of the OpenAI tool list: keep names + parameter schemas, but
// truncate verbose descriptions so the fixed tool overhead fits the local window.
function trimToolsForLocal(tools) {
  if (!LOCAL_TOOL_TRIM || !Array.isArray(tools)) return tools;
  const max = LOCAL_TOOL_DESC_MAX;
  return tools.map(t => {
    if (!t || typeof t !== 'object') return t;
    const nt = Object.assign({}, t);
    if (nt.function && typeof nt.function === 'object') {
      const f = Object.assign({}, nt.function);
      if (typeof f.description === 'string' && f.description.length > max) f.description = f.description.slice(0, max) + '…';
      if (f.parameters && typeof f.parameters === 'object') f.parameters = capSchemaDescriptions(f.parameters, max);
      nt.function = f;
    }
    return nt;
  });
}

// Sum the plain text of an array of messages (used to size compaction chunks).
function sumContent(msgs) {
  let s = '';
  for (const m of msgs) s += contentText(m.content);
  return s;
}

// Ask the local backend to summarize a chunk of turns, optionally merging an existing
// running summary. Returns the summary string, or null if the call failed (caller should
// fall back to dropping the chunk). Talks to Ollama DIRECTLY (not via callUpstream) so it
// never recurses through the proxy itself.
async function localSummarize(chunk, prevSummary, summarizerModel) {
  const sys = '你是一个对话压缩助手。请把下面的对话片段压缩为简洁、信息密集的中文摘要，保留：关键事实、决定、用户意图、待办事项、以及任何对后续对话重要的上下文。若提供了已有摘要，请把新片段融合进去，输出一份更新后的完整摘要，不要重复、不要解释你的步骤。只输出摘要本身，不要加任何前缀。';
  let userText = '';
  if (prevSummary) userText += '【已有的摘要】\n' + prevSummary + '\n\n';
  userText += '【需要压缩的对话片段】\n';
  for (const m of chunk) {
    const role = m.role === 'user' ? '用户' : (m.role === 'assistant' ? '助手' : m.role);
    userText += role + '：' + contentText(m.content) + '\n';
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT);
  try {
    const r = await fetch(LOCAL_BASE_URL + '/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + LOCAL_API_KEY },
      body: JSON.stringify({ model: summarizerModel, messages: [{ role: 'system', content: sys }, { role: 'user', content: userText }], stream: false, max_tokens: 1024 }),
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!r.ok) { log('localSummarize: upstream ' + r.status); return null; }
    const j = await r.json();
    const msg = (j.choices && j.choices[0] && j.choices[0].message) || null;
    if (!msg) return null;
    let t = msg.content || msg.reasoning || msg.reasoning_content || '';
    if (typeof t !== 'string') t = '';
    return t.trim() || null;
  } catch (e) {
    clearTimeout(timer);
    log('localSummarize error: ' + (e && e.message));
    return null;
  }
}

// Recursive summary-based compaction. Loops: grab oldest turns into a chunk, summarize them
// (merging the running digest), replace them with the digest, repeat until the whole prompt
// fits `limit` (or we hit MAX rounds). If summarization fails on a chunk, that chunk is simply
// dropped (lossy fallback). Returns the new message list + telemetry.
async function summarizeCompaction(messages, limit, overhead) {
  const sys = [];
  const rest = [];
  for (const m of messages) {
    if (sys.length === 0 && m.role === 'system') sys.push(m);
    else rest.push(m);
  }
  const summarizerModel = LOCAL_SUMMARIZER || LOCAL_MODEL;
  const chunkBudget = Math.floor(limit * 0.7); // leave room for instruction + prev summary in the summarizer call
  let summary = '';
  let rounds = 0;
  while (estimateTokens(sys) + estimateTokens(rest) + estimateTextTokens(summary) + overhead > limit && rest.length > 1 && rounds < LOCAL_SUMMARIZE_ROUNDS) {
    const chunk = [];
    // Grow the chunk with oldest turns until it approaches chunkBudget (the summarizer call
    // must itself fit within the local window).
    while (rest.length > 1 && estimateTextTokens(sumContent(chunk) + contentText(rest[0].content)) <= chunkBudget) {
      chunk.push(rest.shift());
    }
    if (!chunk.length) chunk.push(rest.shift()); // safety: always make progress
    const sumText = await localSummarize(chunk, summary, summarizerModel);
    if (sumText) summary = sumText; // null => this chunk is dropped (lossy fallback)
    rounds++;
  }
  const out = sys.slice();
  if (summary) out.push({ role: 'user', content: '【以下是较早对话的压缩摘要，供参考】\n' + summary });
  for (const m of rest) out.push(m);
  return { messages: out, summary, rounds, dropped: messages.length - out.length };
}

function buildOpenAIMessages(body) {
  const out = [];
  let systemText = '';
  if (typeof body.system === 'string') systemText = body.system;
  else if (Array.isArray(body.system)) {
    for (const b of body.system) if (b && b.type === 'text') systemText += b.text;
  }
  if (systemText) out.push({ role: 'system', content: systemText });

  const msgs = body.messages || [];
  for (const m of msgs) {
    const role = m.role;
    const content = m.content;
    if (role === 'user') {
      let text = '';
      const toolResults = [];
      if (typeof content === 'string') text = content;
      else if (Array.isArray(content)) {
        for (const b of content) {
          if (!b) continue;
          if (b.type === 'text') text += b.text;
          else if (b.type === 'tool_result') {
            let tc = '';
            if (typeof b.content === 'string') tc = b.content;
            else if (Array.isArray(b.content)) tc = b.content.map(x => (x.type === 'text' ? x.text : (x.type === 'image' ? '[image]' : ''))).join('');
            toolResults.push({ role: 'tool', tool_call_id: b.tool_use_id, content: tc });
          }
        }
      }
      for (const tr of toolResults) out.push(tr);
      if (text) out.push({ role: 'user', content: text });
    } else if (role === 'assistant') {
      let text = '';
      const toolCalls = [];
      if (typeof content === 'string') text = content;
      else if (Array.isArray(content)) {
        for (const b of content) {
          if (!b) continue;
          if (b.type === 'text') text += b.text;
          else if (b.type === 'tool_use') {
            toolCalls.push({ id: b.id, type: 'function', function: { name: b.name, arguments: JSON.stringify(b.input || {}) } });
          }
        }
      }
      const am = { role: 'assistant' };
      if (toolCalls.length) am.tool_calls = toolCalls;
      am.content = toolCalls.length ? (text || null) : text;
      out.push(am);
    }
  }
  return out;
}

function buildOpenAITools(tools) {
  if (!Array.isArray(tools)) return undefined;
  return tools.map(t => ({
    type: 'function',
    function: { name: t.name, description: t.description || '', parameters: t.input_schema || {} }
  }));
}

function openaiToAnthropic(j, model) {
  const msg = j.choices && j.choices[0] && j.choices[0].message ? j.choices[0].message : {};
  const content = [];
  const hasTools = Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0;
  if (msg.content) {
    content.push({ type: 'text', text: msg.content });
  } else if (!hasTools) {
    // Some local GGUF builds (e.g. Qwythos-9B) ship a chat template that never
    // closes the <think> block, so Ollama routes the whole answer -- reasoning
    // AND the final text -- into `reasoning` and leaves `content` empty.
    // Without this fallback Claude Code would receive an empty reply.
    const reasoning = msg.reasoning || msg.reasoning_content;
    if (reasoning) content.push({ type: 'text', text: reasoning });
  }
  if (Array.isArray(msg.tool_calls)) {
    for (const tc of msg.tool_calls) {
      let input = {};
      try { input = JSON.parse(tc.function && tc.function.arguments ? tc.function.arguments : '{}'); } catch (e) { input = {}; }
      content.push({ type: 'tool_use', id: tc.id, name: tc.function ? tc.function.name : '', input });
    }
  }
  const stopReason = (Array.isArray(msg.tool_calls) && msg.tool_calls.length) ? 'tool_use' : 'end_turn';
  const u = j.usage || {};
  return {
    id: 'msg_' + (j.id || Date.now()),
    type: 'message', role: 'assistant', model: model || FORCE_MODEL,
    content, stop_reason: stopReason, stop_sequence: null,
    usage: { input_tokens: u.prompt_tokens || 0, output_tokens: u.completion_tokens || 0 }
  };
}

function sendError(res, status, msg) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ type: 'error', error: { type: 'invalid_request_error', message: msg } }));
}

// Generic upstream caller for ANY OpenAI-compatible endpoint (NVIDIA cloud or a
// local server). Tries the requested model, then falls back through `chain` on
// 503/429/timeout exhaustion. `chain` is supplied by the caller so local calls can
// use a single-entry chain (no cloud fallback) while NVIDIA calls get the full pool.
async function callUpstream(baseUrl, apiKey, payload, stream, res, effectiveModel, chain, isLocal) {
  const url = baseUrl + '/chat/completions';
  const MAX_RETRIES = 3;
  const t0 = Date.now();
  let lastStatus = null, lastText = '';

  for (let ci = 0; ci < chain.length; ci++) {
    const model = chain[ci];
    const attemptPayload = Object.assign({}, payload, { model });
    let served = false;
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT);
      try {
        const r = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + apiKey },
          body: JSON.stringify(attemptPayload),
          signal: controller.signal,
        });
        clearTimeout(timer);
        if (r.ok) {
          if (!stream) {
            const j = await r.json();
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(openaiToAnthropic(j, model)));
            served = true;
            logUsage(model, j.usage, Date.now() - t0, 'ok');
            return;
          }
          // Streaming: write SSE headers then translate the OpenAI stream.
          res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' });
          res.write('event: message_start\n');
          res.write('data: ' + JSON.stringify({
            type: 'message_start',
            message: { id: 'msg_' + Date.now(), type: 'message', role: 'assistant', model: model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } }
          }) + '\n\n');

          let textStarted = false;
          const toolCalls = [];
          // Fallback buffer for local models whose template leaks the answer
          // into `reasoning` and never emits `content` (see openaiToAnthropic).
          let reasoningBuf = '';
          let sawTool = false;
          const usage = { input_tokens: 0, output_tokens: 0 };
          const reader = r.body.getReader();
          const decoder = new TextDecoder();
          let buf = '';
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            let idx;
            while ((idx = buf.indexOf('\n')) >= 0) {
              let line = buf.slice(0, idx).replace(/\r$/, '');
              buf = buf.slice(idx + 1);
              if (!line.startsWith('data:')) continue;
              let data = line.slice(5).trim();
              if (data === '[DONE]') continue;
              let chunk; try { chunk = JSON.parse(data); } catch (e) { continue; }
              const delta = (chunk.choices && chunk.choices[0]) ? chunk.choices[0].delta : {};
              if (delta.content) {
                if (!textStarted) {
                  textStarted = true;
                  res.write('event: content_block_start\n');
                  res.write('data: ' + JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }) + '\n\n');
                }
                res.write('event: content_block_delta\n');
                res.write('data: ' + JSON.stringify({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: delta.content } }) + '\n\n');
              } else if (delta.reasoning || delta.reasoning_content) {
                reasoningBuf += (delta.reasoning || delta.reasoning_content);
              }
              if (Array.isArray(delta.tool_calls)) {
                for (const tc of delta.tool_calls) {
                  const i = (typeof tc.index === 'number') ? tc.index : 0;
                  if (!toolCalls[i]) toolCalls[i] = { index: i, id: tc.id || ('call_' + i), name: (tc.function && tc.function.name) || '', args: '' };
                  if (tc.id) toolCalls[i].id = tc.id;
                  if (tc.function && tc.function.name) toolCalls[i].name = tc.function.name;
                  if (tc.function && tc.function.arguments) toolCalls[i].args += tc.function.arguments;
                }
                sawTool = true;
              }
              if (chunk.usage) { usage.input_tokens = chunk.usage.prompt_tokens || 0; usage.output_tokens = chunk.usage.completion_tokens || 0; }
            }
          }
          // No text and no tools came through, but we buffered reasoning:
          // flush it as the answer so the client never sees an empty reply.
          if (!textStarted && !toolCalls.length && reasoningBuf) {
            textStarted = true;
            res.write('event: content_block_start\n');
            res.write('data: ' + JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }) + '\n\n');
            res.write('event: content_block_delta\n');
            res.write('data: ' + JSON.stringify({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: reasoningBuf } }) + '\n\n');
          }
          if (textStarted) {
            res.write('event: content_block_stop\n');
            res.write('data: ' + JSON.stringify({ type: 'content_block_stop', index: 0 }) + '\n\n');
          }
          let tIndex = textStarted ? 1 : 0;
          for (const tc of toolCalls) {
            if (!tc) continue;
            res.write('event: content_block_start\n');
            res.write('data: ' + JSON.stringify({ type: 'content_block_start', index: tIndex, content_block: { type: 'tool_use', id: tc.id, name: tc.name } }) + '\n\n');
            res.write('event: content_block_delta\n');
            res.write('data: ' + JSON.stringify({ type: 'content_block_delta', index: tIndex, delta: { type: 'input_json_delta', partial_json: tc.args } }) + '\n\n');
            res.write('event: content_block_stop\n');
            res.write('data: ' + JSON.stringify({ type: 'content_block_stop', index: tIndex }) + '\n\n');
            tIndex++;
          }
          const stopReason = sawTool ? 'tool_use' : 'end_turn';
          res.write('event: message_delta\n');
          res.write('data: ' + JSON.stringify({ type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { input_tokens: usage.input_tokens, output_tokens: usage.output_tokens } }) + '\n\n');
          res.write('event: message_stop\n');
          res.write('data: ' + JSON.stringify({ type: 'message_stop' }) + '\n\n');
          res.end();
          served = true;
          logUsage(model, usage, Date.now() - t0, 'ok');
          return;
        }
        const txt = await r.text();
        lastStatus = r.status; lastText = txt;
        if ((r.status === 503 || r.status === 429) && attempt < MAX_RETRIES) {
          const delay = Math.min(2000 * attempt, 8000);
          log('upstream ' + r.status + ' (model=' + model + ' attempt ' + attempt + '/' + MAX_RETRIES + '), retrying in ' + delay + 'ms');
          await new Promise(r => setTimeout(r, delay));
          continue;
        }
        if (ci < chain.length - 1) {
          log('model ' + model + ' exhausted (' + r.status + '); falling back to ' + chain[ci + 1]);
          break;
        }
        // Local context-window overflow: hand back a sentinel so the caller can re-compact
        // and retry instead of the proxy committing a 400 to Claude Code.
        if (isLocal && r.status === 400 && /exceed_context_size_error/.test(txt)) {
          log('local context exceeded (n_ctx hit, ' + (r.status) + '); signaling caller to re-compact');
          return LOCAL_CONTEXT_EXCEEDED;
        }
        log('upstream error', r.status, txt.slice(0, 500));
        sendError(res, r.status, 'upstream error: ' + txt.slice(0, 800));
        return;
      } catch (e) {
        clearTimeout(timer);
        lastStatus = 'ERR'; lastText = e && e.message;
        if (e && e.name === 'AbortError' && attempt < MAX_RETRIES) {
          log('upstream timeout (model=' + model + ' attempt ' + attempt + '/' + MAX_RETRIES + '), retrying...');
          await new Promise(r => setTimeout(r, 2000));
          continue;
        }
        if (attempt < MAX_RETRIES) {
          log('upstream error (model=' + model + ' attempt ' + attempt + '/' + MAX_RETRIES + '): ' + (e && e.message) + ', retrying...');
          await new Promise(r => setTimeout(r, 2000));
          continue;
        }
        if (ci < chain.length - 1) {
          log('model ' + model + ' failed (' + (e && e.message) + '); falling back to ' + chain[ci + 1]);
          break;
        }
        log('proxy error', e && e.message);
        if (!res.headersSent) sendError(res, 500, 'proxy error: ' + (e && e.message));
        else { try { res.end(); } catch (_) {} }
        return;
      }
    }
    if (served) return;
  }
  log('all models in fallback chain failed (last=' + lastStatus + ')');
  if (!res.headersSent) sendError(res, 502, 'all NVIDIA models unavailable: ' + lastText.slice(0, 400));
  else { try { res.end(); } catch (_) {} }
}

function aggregateStats() {
  const file = path.join(__dirname, 'usage.jsonl');
  const out = { total: 0, by_model: {}, last: null };
  try {
    const lines = fs.readFileSync(file, 'utf8').split('\n').filter(Boolean);
    for (const ln of lines) {
      let e; try { e = JSON.parse(ln); } catch (_) { continue; }
      out.total++;
      out.by_model[e.model] = out.by_model[e.model] || { calls: 0, prompt_tokens: 0, completion_tokens: 0 };
      out.by_model[e.model].calls++;
      out.by_model[e.model].prompt_tokens += e.prompt_tokens || 0;
      out.by_model[e.model].completion_tokens += e.completion_tokens || 0;
      out.last = e;
    }
  } catch (e) {}
  return out;
}

const server = http.createServer(async (req, res) => {
  const urlPath = (req.url || '').split('?')[0];
  if (req.method === 'GET' && urlPath === '/v1/models') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    const data = MODELS.map(m => ({
      id: m.id, type: 'model', display_name: m.display_name,
      created_at: 1785145588495, owned_by: 'nvidia',
    }));
    if (LOCAL_MODEL) data.push({ id: 'local/' + LOCAL_MODEL, type: 'model', display_name: 'Local ' + LOCAL_MODEL + ' (Ollama)', created_at: 1785145588495, owned_by: 'local' });
    res.end(JSON.stringify({ object: 'list', data }));
    return;
  }
  if (req.method === 'GET' && urlPath === '/v1/stats') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(aggregateStats()));
    return;
  }
  if (req.method === 'POST' && urlPath === '/v1/messages') {
    let raw = '';
    req.on('data', c => raw += c);
    req.on('end', async () => {
      let body;
      try { body = JSON.parse(raw); } catch (e) { return sendError(res, 400, 'invalid JSON body'); }
      log('request model=' + (body.model || '?') + ' stream=' + !!body.stream + ' tools=' + (Array.isArray(body.tools) ? body.tools.length : 0));
      const oaMessages = buildOpenAIMessages(body);
      const oaTools = buildOpenAITools(body.tools);
      const stream = !!body.stream;
      const effectiveModel = resolveModel(body.model);
      log('effective model=' + effectiveModel + ' (requested=' + (body.model || '?') + ')');
      const payload = { model: effectiveModel, messages: oaMessages, stream, max_tokens: body.max_tokens || 1024 };
      if (oaTools) payload.tools = oaTools;
      if (typeof body.temperature === 'number') payload.temperature = body.temperature;
      if (typeof body.top_p === 'number') payload.top_p = body.top_p;
      if (typeof body.top_k === 'number') payload.top_k = body.top_k;
      if (typeof body.presence_penalty === 'number') payload.presence_penalty = body.presence_penalty;
      if (typeof body.frequency_penalty === 'number') payload.frequency_penalty = body.frequency_penalty;
      if (Array.isArray(body.stop_sequences)) payload.stop = body.stop_sequences;

      // --- local / cloud routing ---
      // Default (hands-off): haiku-tier -> local, sonnet/opus -> NVIDIA (via TIER_MODELS).
      // Manual override: a leading "@local" / "@nvidia" token in the last user message
      // forces that one request to the chosen backend (marker stripped before send).
      const override = detectOverride(oaMessages);
      const isLocalTier = typeof effectiveModel === 'string' && effectiveModel.startsWith('local/');
      let useLocal = isLocalTier;
      if (override === 'local') useLocal = true;
      else if (override === 'nvidia') useLocal = false;

      if (useLocal) {
        const tag = isLocalTier ? effectiveModel.slice('local/'.length) : LOCAL_MODEL;
        // Estimate the fixed overhead Ollama charges but compaction CANNOT drop: the tool
        // schemas (Claude Code sends its whole tool set) + chat-template special tokens.
        // Only the conversation turns themselves are trimmable.
        // IMPORTANT: for local routes, shrink (or drop) the tool definitions so the fixed tool
        // overhead fits the small local window. Tool names + schemas stay intact when trimming.
        const localTools = LOCAL_DROP_TOOLS ? null : trimToolsForLocal(oaTools);
        const toolOverheadRaw = estimateTextTokens(JSON.stringify(oaTools || ''));
        const toolOverhead = estimateTextTokens(JSON.stringify(localTools || ''));
        const templateOverhead = 64 + 12 * oaMessages.length;
        const overhead = toolOverhead + templateOverhead;
        if (LOCAL_DROP_TOOLS) log('local tools DROPPED (pure-chat local session; ' + (oaTools ? oaTools.length : 0) + ' tools omitted)');
        else if (toolOverheadRaw - toolOverhead > 0) log('local tool-trim: overhead ' + toolOverheadRaw + ' -> ' + toolOverhead + ' tok (' + (oaTools ? oaTools.length : 0) + ' tools, cap=' + LOCAL_TOOL_DESC_MAX + ')');
        // Safe prompt ceiling: leave room for the reply and a margin for estimate error.
        // Derive from LOCAL_REPLY_CAP (NOT the raw body.max_tokens — Claude Code sends a huge
        // value like 32000 for the local model, which would collapse the limit below system+tools).
        const replyCap = Math.min(body.max_tokens || 1024, LOCAL_REPLY_CAP);
        const genReserve = replyCap;
        const hardCap = (LOCAL_MAX_PROMPT && LOCAL_MAX_PROMPT > 0) ? LOCAL_MAX_PROMPT : Infinity;
        const limit = Math.min(Math.floor((LOCAL_NUM_CTX - genReserve) * 0.8), hardCap);

        let localMessages = oaMessages;
        let compactNote = '';
        // Stage 1: recursive SUMMARY compaction (the auto-"/compact" behavior) — preserves
        // context by digesting old turns into a running summary instead of discarding them.
        // Loops until the prompt fits the local window.
        if (LOCAL_SUMMARIZE && estimateTokens(oaMessages) + overhead > limit) {
          const sc = await summarizeCompaction(oaMessages, limit, overhead);
          localMessages = sc.messages;
          compactNote = ', summary-compacted (' + sc.rounds + ' round(s), folded ' + sc.dropped + ' turns into a digest)';
          log('local summary-compact: ' + sc.rounds + ' round(s); est ' + estimateTokens(oaMessages) + ' -> ' + estimateTokens(localMessages) + ' (+overhead ' + overhead + ', limit ' + limit + ')');
        }
        // Stage 2: if summary compaction couldn't bring it under (or is disabled), fall back
        // to dropping oldest turns — still keeps the most recent context.
        if (estimateTokens(localMessages) + overhead > limit) {
          const cmp = compactLocalMessages(localMessages, limit, overhead);
          if (cmp.dropped > 0) {
            localMessages = cmp.messages;
            compactNote += ', then dropped ' + cmp.dropped + ' more oldest turn(s)';
            log('local auto-compact (drop fallback): dropped ' + cmp.dropped + ' oldest turn(s); est -> ' + estimateTokens(localMessages));
          }
        }
        const localPayload = Object.assign({}, payload, { model: tag, messages: localMessages, max_tokens: replyCap });
        if (localTools && localTools.length) localPayload.tools = localTools; else delete localPayload.tools;
        if (!localPayload.options) localPayload.options = {};
        if (!localPayload.options.num_ctx) localPayload.options.num_ctx = LOCAL_NUM_CTX;
        log('route -> LOCAL (' + tag + ', num_ctx=' + localPayload.options.num_ctx + ')' + compactNote + (override ? ' [override=' + override + ']' : ''));
        let localResult = await callUpstream(LOCAL_BASE_URL, LOCAL_API_KEY, localPayload, stream, res, 'local:' + tag, [tag], true);
        // Safety net: if the first compact STILL overflowed (estimate error on a pathological
        // prompt), drop to the minimal survivable set (system + last turn) and retry ONCE.
        // If that too overflows, the prompt is genuinely too big for the local window — report
        // an actionable error instead of looping forever.
        if (localResult === LOCAL_CONTEXT_EXCEEDED) {
          const minimal = minimalLocalMessages(oaMessages);
          log('local still exceeded after compact; retrying with minimal set (est ' + estimateTokens(minimal) + ')');
          const minPayload = Object.assign({}, payload, { model: tag, messages: minimal, max_tokens: replyCap });
          if (localTools && localTools.length) minPayload.tools = localTools; else delete minPayload.tools;
          if (!minPayload.options) minPayload.options = {};
          minPayload.options.num_ctx = LOCAL_NUM_CTX;
          localResult = await callUpstream(LOCAL_BASE_URL, LOCAL_API_KEY, minPayload, stream, res, 'local:' + tag, [tag], true);
          if (localResult === LOCAL_CONTEXT_EXCEEDED) {
            sendError(res, 400, 'local context window (' + LOCAL_NUM_CTX + ' tok) exceeded even at the minimal prompt. The system prompt + tool definitions + current message alone exceed the local model\'s window. Switch this session to a cloud model, or raise localNumCtx in Modelfile.qwythos and recreate the Ollama model for more room.');
          }
        }
      } else {
        const nvModel = isLocalTier ? FORCE_MODEL : effectiveModel; // override forced cloud but tier was local
        const nvPayload = Object.assign({}, payload, { model: nvModel });
        log('route -> NVIDIA (' + nvModel + ')' + (override ? ' [override=' + override + ']' : ''));
        callUpstream(NVIDIA_BASE, NVIDIA_KEY, nvPayload, stream, res, nvModel, fallbackChain(nvModel));
      }
    });
    return;
  }
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'not found' }));
});

fetchCatalog();
server.listen(PORT, HOST, () => log('router proxy listening on http://' + HOST + ':' + PORT + '  cloud=' + FORCE_MODEL + '  local=' + LOCAL_BASE_URL + '  upstreamTimeout=' + UPSTREAM_TIMEOUT + 'ms'));
server.on('error', (e) => {
  if (e.code === 'EADDRINUSE') { log('Port ' + PORT + ' already in use; assuming proxy already running.'); process.exit(0); }
  log('listen error', e.message); process.exit(1);
});
