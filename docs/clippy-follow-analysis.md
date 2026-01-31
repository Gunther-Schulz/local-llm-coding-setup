# Why the model didn’t follow CLIPPY well – log analysis

From `logs/proxy.log` we can see **two proxy behaviors** that directly explain why CLIPPY wasn’t followed: tool condense and sliding window. Model capability may also play a role, but the main issue is that the model **never had** the full CLIPPY protocol.

---

## 1. Tool condense: model never saw full CLIPPY (main cause)

**What happens**

- CLIPPY_MKII.md is **1191 lines** (~tens of thousands of characters).
- Proxy config: `TOOL_RESPONSE_MAX_VERBATIM=2000`, `TOOL_RESPONSE_PREVIEW_CHARS=500`.
- Any tool response **over 2000 chars** is replaced by a **500‑character preview** on every request.

**In the log**

- First request where the model had already called `Read(CLIPPY_MKII.md)`: e.g. `messages_in: 5`, **`tool_condense: 1 response(s) condensed`**, `tokens_after_condense: 19831`.
- So the **Read(CLIPPY)** result was condensed to a 500‑char preview before being sent to the backend.

**Effect**

- The model only ever saw **~500 characters** of CLIPPY (opening lines + “… N more characters omitted …”).
- It never had the full protocol: A1.1 / A1.2 / A1.3, verification rules, trackers, menus, checklists, etc.
- Following CLIPPY “properly” requires the full doc; with only a short preview, the model is effectively guessing.

**Conclusion:** **Tool condense is the main reason** the model didn’t follow CLIPPY: it never had the full instructions.

---

## 2. Sliding window: task and CLIPPY reference were summarized away

**What happens**

- When `messages > 50` (or tokens > threshold), we keep only the **last 40** conversation messages.
- Older messages are **summarized** into 2–3 sentences and sent as one user message.

**In the log**

- Multiple requests show e.g.:
  - `sliding_window: yes (trigger: messages>50)`
  - `messages_summarized: 10` … then 12, 14, 16, 18, 20, 22…
  - `Generated summary: The user is working on refactoring a codebase using Clippy as a guide, specifically focusi...`

So the **oldest** messages (including the user’s “@CLIPPY_MKII.md use clipy for this task” and the condensed CLIPPY read) were **replaced** by that short summary.

**Effect**

- Later in the conversation the model no longer saw:
  - The explicit instruction to “use CLIPPY”
  - The 500‑char CLIPPY preview
- It only saw a generic summary like “refactoring using Clippy as a guide”.
- So even the **reference** to CLIPPY and the task was diluted.

**Conclusion:** **Sliding window is a contributing factor**: once the conversation grew, the model lost the clear “use CLIPPY” context and the only CLIPPY content it had (the preview).

---

## 3. Model capability (possible, but secondary)

- A 30B model may still not follow a long, strict protocol as reliably as a frontier model.
- From the log we **cannot** tell how well it would have followed CLIPPY with **full** context, because it never had it: first tool condense, then sliding window removed the rest.

So “model not good enough” is possible but **secondary** to the two proxy effects above.

---

## Recommended mitigations

**1. Don’t condense “instruction” tool results (e.g. CLIPPY)**

- For `Read` (or similar) of paths like `*CLIPPY*.md`, `*RULE*.md`, or a short allowlist, **skip** condense (keep full content), so the model actually sees the full protocol.
- Implement by: path allowlist, or tool name + path pattern, and bypass `condense_large_tool_response` for those messages.

**2. Increase verbatim/preview for “important” reads**

- Alternatively (or in addition): raise `TOOL_RESPONSE_MAX_VERBATIM` or `TOOL_RESPONSE_PREVIEW_CHARS` when the tool is Read and path matches an instruction doc, so at least a much larger chunk of CLIPPY is sent (e.g. first 8k chars instead of 500).

**3. Keep the user request out of the summarization window**

- When building the “summary” message, **prepend** the original user instruction (e.g. first user message of the conversation) to the summary, so “use CLIPPY to refactor” (and similar) is always in context even when sliding window is active.
- Or: never summarize the first user message; always keep it verbatim in the prompt.

**4. Optional: larger window or higher trigger**

- Increasing `CONTEXT_WINDOW_SIZE` or `COMPRESSION_TRIGGER_MESSAGES` delays when the CLIPPY instruction and the condensed preview fall into “summarized” and get replaced. It doesn’t fix tool condense but reduces the impact of sliding window.

---

## Summary

| Cause              | Effect |
|--------------------|--------|
| **Tool condense**  | Model only ever saw ~500 chars of CLIPPY; never had full protocol. **Main reason** it didn’t follow CLIPPY. |
| **Sliding window** | After 50+ messages, original “use CLIPPY” and the 500‑char preview were summarized away, so the model lost the task and the only CLIPPY text it had. |
| **Model**          | May still matter, but we can’t assess it until the model gets full (or much larger) CLIPPY and the user instruction is kept in context. |

So from the log we can say: **the main reason the model didn’t follow CLIPPY well is that our proxy never gave it the full CLIPPY doc (tool condense) and later removed the only CLIPPY reference and the explicit task (sliding window).** Fixing condense for instruction docs and preserving the user request in the window should help a lot.
