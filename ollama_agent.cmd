@echo off
REM ollama_agent launcher: run the local multi-agent orchestrator from anywhere.
REM E:\claude-nvidia is on PATH, so `ollama_agent ...` resolves to this file.
python "%~dp0ollama_agent.py" %*
