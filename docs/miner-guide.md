# Agent development

This repository is **local benchmark only**. For writing and testing agents, see the root [README.md](../README.md): run `docker compose --profile test run test --agent-file …` with `OPENROUTER_API_KEY` and/or `CHUTES_API_KEY` set.

Agent code lives under `src/agent/`; the default interface expected by the sandbox is documented in `src/agent/agent_interface.py` and the sample agents `agent.py` / `agent_test.py`.
