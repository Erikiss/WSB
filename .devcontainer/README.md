# WSBench in a GitHub Codespace

This folder lets you run WSBench in a private Codespace and open it from a tablet over HTTPS.

1. **Secret (once).** On GitHub: Settings → Codespaces → Secrets → New secret, name `OPENROUTER_API_KEY`, repository access: this repository. Without it the app still runs, and Assist reports `no-key`.
   Optional second secret `ANTHROPIC_API_KEY` (from the Claude Console): with it, the Claude models (planner, report and translate) are billed to your Anthropic credits, and OpenRouter is used only for the other models. After adding a secret to an existing Codespace, stop and restart it. The header then shows `Claude direct yes`. To test one call from the Codespace terminal:
   `python -c "from wsbench.assist import openrouter as o; r=o.chat('translate','anthropic/claude-fable-5.2','Answer in one word.',[{'role':'user','content':'Say ok.'}]); print(r['provider'], r['ok'], r['error'], r['cost_usd'])"`
2. **Create the Codespace.** On the repository page: Code → Codespaces → Create codespace. The first start installs everything and downloads the model and the ARC data (a few minutes).
3. **Open the app.** In the Codespace, open the Ports panel and click the globe next to port 8765 ("WSBench"). The address looks like `https://<name>-8765.app.github.dev`. Keep the port visibility on **Private**: the app has no login of its own, and only your GitHub account can then open the link.
4. **Check.** The header should show `LLM live`. If it shows `no-key`, the secret was added after the Codespace was created: stop and restart the Codespace, or run `bash .devcontainer/start.sh` in its terminal.

The server restarts on every Codespace start. Its log is `/tmp/wsbench.log`. Stop the Codespace when you are done, so it does not use up your free hours.

If your API key does not belong to the Default workspace of your Anthropic organisation, set the environment variable `WSB_ANTHROPIC_WORKSPACE` to that workspace's ID (visible in the Console URL), so the link to the session in the Agent tab opens the right page.
