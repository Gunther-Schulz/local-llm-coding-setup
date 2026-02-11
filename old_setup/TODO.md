Here’s a concise plan for when you do it later.
1. ~~Removing the custom tool-call transformation~~ (done)
Safe to remove (once llama-server is your only backend and you’ve confirmed it returns native tool_calls for Qwen):
proxy/tool_parser.py: parse_qwen_tool_calls, should_transform_tool_calls, generate_tool_call_chunks, and any “transform response” helpers.
In proxy/streaming.py: the branch that, at stream end, parses content and emits tool-call chunks when the backend didn’t send tool_calls (and the state used only for that).
In proxy/server.py: any non-streaming path that runs the same “parse content → inject tool_calls” logic.
The tool-service layer that exists only to support this transformation.
Do not remove: Anything that adds the virtual tool to the request, or that intercepts and fulfills that tool call and injects its result. That’s separate from “transform raw text → tool_calls.”
2. ~~Removing vLLM as a backend~~ (done)
vLLM support removed: run/vllm.py and run/select_engine.py deleted; run/run and run/llm.py use llama-server only; setup/install.sh no longer installs vLLM or PyTorch; config/models.conf tool_parser kept for docs.
3. Keeping virtual tool injection working
Virtual tool behavior is independent of:
the custom transformation (we only stop “parsing text → tool_calls”),
and the vLLM backend (we only stop starting vLLM and wiring it into the run/engine selection).
Must keep intact:
Add virtual tool to request
When get_stored(conversation_id) and we’re building backend_request["tools"], add VIRTUAL_TOOL_DEFINITION (e.g. search_compressed_conversation) to that list.
Tool choice when only virtual tool
When the client didn’t send tools and we add only the virtual tool, set backend_request["tool_choice"] = "auto" so the model can call it.
Intercept and inject result
_inject_virtual_tool_results: before sending to the backend, if the last assistant message contains a call to search_compressed_conversation and there’s no corresponding tool result yet, call execute_virtual_tool(conversation_id, args), then append a role: "tool" message (with name=VIRTUAL_TOOL_NAME and the result) to the message list.
Context manager
store_compressed, get_stored, execute_virtual_tool, and the virtual tool name/definition in proxy/context_manager.py.
When you delete transformation/vLLM code, don’t remove or refactor the blocks that do the above four things; treat “virtual tool injection” as a separate feature that stays.
4. Order of operations when you do it
~~Confirm llama-server returns native tool_calls~~ (done; --jinja enabled). ~~Remove the transformation code~~ (done).
~~Remove vLLM~~ (done).
After each step, run a flow that uses the virtual tool (e.g. trigger compression, then a turn that calls search_compressed_conversation) and confirm the proxy still injects the tool result and the backend receives it.
So: yes, you can remove the custom transformation (and vLLM) later; just be careful to keep every part that adds the virtual tool to the request and that runs _inject_virtual_tool_results / execute_virtual_tool and injects the tool message—that’s what keeps the virtual tool working.