// free-claude-proxy.js
// Generic Anthropic -> OpenAI translation proxy for Claude Code.
// Lets any OpenAI-compatible "free" LLM provider (Zhipu / Ali / Volcano /
// NVIDIA / SiliconFlow / Tencent) be used by Claude Code, which only speaks
// the Anthropic Messages API. Zero npm dependencies (Node built-ins only).
//
// Configured entirely via environment variables (set by free.bat):
//   UPSTREAM_BASE_URL  e.g. https://api.siliconflow.cn/v1
//   UPSTREAM_API_KEY   provider API key (injected here; Claude Code's key is ignored)
//   UPSTREAM_MODEL     default model id when Claude Code sends claude-* or nothing
//   UPSTREAM_MODELS    comma-separated free-model ids (served at /v1/models)
//   PORT               listen port (default 3457)
//   HOST               listen host (default 127.0.0.1)
//   UPSTREAM_TIMEOUT   upstream request timeout ms (default 120000)

const http = require('http');
const fs = require('fs');
const path = require('path');

const CONFIG_PATH = process.env.FREE_PROXY_CONFIG ? path.resolve(process.env.FREE_PROXY_CONFIG) : null;
let cfg = {};
if (CONFIG_PATH) {
  try { cfg = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')); } catch (e) { cfg = {}; }
}

const UPSTREAM_BASE_URL = (process.env.UPSTREAM_BASE_URL || cfg.upstreamBaseUrl || '').replace(/\/+$/, '');
const UPSTREAM_API_KEY = process.env.UPSTREAM_API_KEY || cfg.upstreamApiKey || '';
const UPSTREAM_MODEL = process.env.UPSTREAM_MODEL || cfg.upstreamModel || '';
const UPSTREAM_MODELS = (process.env.UPSTREAM_MODELS || cfg.upstreamModels || '')
  .split(',').map(s => s.trim()).filter(Boolean);
const HOST = process.env.HOST || cfg.listenHost || '127.0.0.1';
const PORT = parseInt(process.env.PORT || cfg.listenPort || '3457', 10);
const UPSTREAM_TIMEOUT = parseInt(process.env.UPSTREAM_TIMEOUT || cfg.upstreamTimeout || '120000', 10);
const PROVIDER = process.env.UPSTREAM_PROVIDER || cfg.provider || 'free';

function log(...a) {
  const s = '[' + new Date().toISOString() + '] ' + a.join(' ');
  console.log(s);
  try { fs.appendFileSync(path.join(__dirname, 'free-proxy.log'), s + '\n'); } catch (e) {}
}

// Build Anthropic -> OpenAI chat messages.
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
    type: 'message', role: 'assistant', model: model || UPSTREAM_MODEL,
    content, stop_reason: stopReason, stop_sequence: null,
    usage: { input_tokens: u.prompt_tokens || 0, output_tokens: u.completion_tokens || 0 }
  };
}

function sendError(res, status, msg) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ type: 'error', error: { type: 'invalid_request_error', message: msg } }));
}

// Resolve the model Claude Code requests into a real upstream model id.
function resolveModel(requested) {
  if (!requested) return UPSTREAM_MODEL;
  if (/^claude-/i.test(requested)) return UPSTREAM_MODEL;       // built-in claude ids -> our default
  if (UPSTREAM_MODELS.includes(requested)) return requested;    // a free model we advertised
  return requested;                                             // pass through anything else
}

async function callUpstream(payload, stream, res, effectiveModel) {
  const url = UPSTREAM_BASE_URL + '/chat/completions';
  const MAX_RETRIES = 3;
  const t0 = Date.now();
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT);
    try {
      const r = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + UPSTREAM_API_KEY },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timer);
      if (r.ok) {
        if (!stream) {
          const j = await r.json();
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(openaiToAnthropic(j, effectiveModel)));
          log('ok model=' + effectiveModel + ' ' + (Date.now() - t0) + 'ms');
          return;
        }
        res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' });
        res.write('event: message_start\n');
        res.write('data: ' + JSON.stringify({
          type: 'message_start',
          message: { id: 'msg_' + Date.now(), type: 'message', role: 'assistant', model: effectiveModel, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } }
        }) + '\n\n');
        let textStarted = false;
        const toolCalls = [];
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
        log('ok(stream) model=' + effectiveModel + ' ' + (Date.now() - t0) + 'ms');
        return;
      }
      const txt = await r.text();
      if ((r.status === 429 || r.status === 500 || r.status === 502 || r.status === 503 || r.status === 529) && attempt < MAX_RETRIES) {
        const delay = Math.min(2000 * attempt, 8000);
        log('upstream ' + r.status + ' (model=' + effectiveModel + ' attempt ' + attempt + '/' + MAX_RETRIES + '), retrying in ' + delay + 'ms');
        await new Promise(r => setTimeout(r, delay));
        continue;
      }
      log('upstream error', r.status, txt.slice(0, 500));
      sendError(res, r.status, 'upstream error: ' + txt.slice(0, 800));
      return;
    } catch (e) {
      clearTimeout(timer);
      if (e && e.name === 'AbortError' && attempt < MAX_RETRIES) {
        log('upstream timeout (model=' + effectiveModel + ' attempt ' + attempt + '/' + MAX_RETRIES + '), retrying...');
        await new Promise(r => setTimeout(r, 2000));
        continue;
      }
      if (attempt < MAX_RETRIES) {
        log('upstream error (model=' + effectiveModel + ' attempt ' + attempt + '/' + MAX_RETRIES + '): ' + (e && e.message) + ', retrying...');
        await new Promise(r => setTimeout(r, 2000));
        continue;
      }
      log('proxy error', e && e.message);
      if (!res.headersSent) sendError(res, 500, 'proxy error: ' + (e && e.message));
      else { try { res.end(); } catch (_) {} }
      return;
    }
  }
  log('all retries failed for model=' + effectiveModel);
  if (!res.headersSent) sendError(res, 502, 'upstream unavailable');
  else { try { res.end(); } catch (_) {} }
}

const server = http.createServer(async (req, res) => {
  const urlPath = (req.url || '').split('?')[0];
  if (req.method === 'GET' && urlPath === '/v1/models') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    const data = (UPSTREAM_MODELS.length ? UPSTREAM_MODELS : [UPSTREAM_MODEL]).filter(Boolean).map(id => ({
      id, type: 'model', display_name: id, created_at: 1785145588495, owned_by: PROVIDER,
    }));
    res.end(JSON.stringify({ object: 'list', data }));
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
      if (Array.isArray(body.stop_sequences)) payload.stop = body.stop_sequences;
      if (body.thinking && typeof body.thinking === 'object') {
        // Some upstreams accept thinking via a vendor field; pass through best-effort.
        if (typeof body.thinking.budget_tokens === 'number') payload.max_tokens = Math.max(payload.max_tokens, body.thinking.budget_tokens);
      }
      callUpstream(payload, stream, res, effectiveModel);
    });
    return;
  }
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'not found' }));
});

server.listen(PORT, HOST, () => log('free-claude-proxy listening on http://' + HOST + ':' + PORT +
  '  provider=' + PROVIDER + '  model=' + UPSTREAM_MODEL + '  upstream=' + UPSTREAM_BASE_URL + '  timeout=' + UPSTREAM_TIMEOUT + 'ms'));
server.on('error', (e) => {
  if (e.code === 'EADDRINUSE') { log('Port ' + PORT + ' already in use; assuming proxy already running.'); process.exit(0); }
  log('listen error', e.message); process.exit(1);
});
