forgets how t use tools calls properly. does not notice when tings dont work and check its actual usage in teh system prompt. works if i point it out

**Qwen3-Coder re-reads same file chunks (known upstream):** Model repeatedly re-reads the same file (e.g. 20+ times on a 1000-line file) instead of moving on. Reported as model/training issue; also related to tool_choice "auto" vs "required" and parser behavior. See: [QwenLM/qwen-code#66](https://github.com/QwenLM/qwen-code/issues/66), [QwenLM/Qwen3-Coder#480](https://github.com/QwenLM/Qwen3-Coder/issues/480). Workarounds: try tool_choice "required" where possible; limit file/chunk size; or use another model for file-heavy tasks.

grammar problem. unknown if it sllama.cpp or a model issue

server prnblem: Request ID: 81c941d7-b677-497b-aa6c-9de75039c080
{"error":57,"details":{"title":"Provider Error","detail":"We're having trouble connecting to the model provider. This might be temporary - please try again in a moment.","isRetryable":false,"additionalInfo":{},"buttons":[],"planChoices":[]},"isExpected":true}
Provider Error We're having trouble connecting to the model provider. This might be temporary - please try again in a moment.
H4t: Provider Error We're having trouble connecting to the model provider. This might be temporary - please try again in a moment.
    at lpf (vscode-file://vscode-app/usr/share/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:9097:38263)
    at apf (vscode-file://vscode-app/usr/share/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:9097:37251)
    at Cpf (vscode-file://vscode-app/usr/share/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:9098:5088)
    at Tva.run (vscode-file://vscode-app/usr/share/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:9098:9098)
    at async Vyt.runAgentLoop (vscode-file://vscode-app/usr/share/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:34206:57104)
    at async mgc.streamFromAgentBackend (vscode-file://vscode-app/usr/share/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:34255:7835)
    at async mgc.getAgentStreamResponse (vscode-file://vscode-app/usr/share/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:34255:8576)
    at async qTe.submitChatMaybeAbortCurrent (vscode-file://vscode-app/usr/share/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:9180:14965)
    at async Xi (vscode-file://vscode-app/usr/share/cursor/resources/app/out/vs/workbench/workbench.desktop.main.js:33004:3808)