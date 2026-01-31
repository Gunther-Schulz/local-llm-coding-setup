# Caveats: Chat retention & queryable history

The proxy’s **chat retention** feature stores compressed conversation state so the model can query it via the virtual tool `search_compressed_conversation`. That feature has strong limitations; use it with these caveats in mind.

---

## When the virtual tool exists

- **Only after compression.** Queryable history is created only when the proxy has **already run compression** for that conversation (i.e. the prompt would have exceeded the backend limit, so we summarized old messages and stored them). If you never hit the context limit—or compression is disabled—there is no stored conversation to query; the virtual tool either isn’t added or returns *[No compressed conversation stored for this session.]*
- **Only when we have a stored conversation.** The proxy adds the virtual tool to the request only when `get_stored(conversation_id)` returns something. So the model can only “search compressed conversation” in sessions where we’ve previously compressed.

---

## Storage and lifetime

- **In-memory only.** The compressed store lives in the proxy process’s RAM. There is no persistence to disk. **Restarting the proxy clears all stored conversations.**
- **Single process.** If you run multiple proxy instances (e.g. load-balanced), they do **not** share the store. Only the process that performed the compression has that conversation’s data; other processes will have no (or different) data for the same logical chat.
- **LRU eviction.** When `COMPRESSED_STORE_MAX_CONVERSATIONS` is set (e.g. 10), the proxy evicts **least recently used** conversations to stay under the cap. Old or inactive chats lose their queryable history once evicted.
- **One snapshot per conversation.** Each time we compress, we **replace** the stored snapshot for that `conversation_id`. We keep the latest full pre-compression state only, not a full history of every compression turn.

---

## Conversation identity

- **Conversation ID is derived.** The proxy derives `conversation_id` from the **first non-system message** (first 100 characters of its text, then MD5). Implications:
  - If the user **edits or deletes** the first user message, or the client sends a different “first” message, the ID can change and the store lookup will miss (or hit a different slot).
  - **Collision risk:** Two different chats that happen to have the same first 100 characters of the first non-system message will share one store entry; the second will overwrite the first.

---

## What the virtual tool returns

- **Result caps.** Responses from the virtual tool are truncated/capped by:
  - `COMPRESSED_STORE_RESULT_MAX_CHARS` (per result)
  - `COMPRESSED_STORE_SEARCH_TOP_K` (max number of message hits for keyword search)
  - `COMPRESSED_STORE_SEARCH_MAX_CHARS` (total characters for keyword search hits)  
  So the model may not see the full message or all hits; long sections or many hits are intentionally limited to avoid blowing context again.
- **Section extraction is heuristic.** Section lookup (e.g. `primary_request`, `key_concepts`) parses the **summary text** using exact header match, then fuzzy match. If the summarization LLM didn’t follow the exact section titles, the tool can return *[Section 'X' not found in summary.]*
- **Keyword search is simple.** Query uses a **case-insensitive substring** match over the plain text of each message. There is no semantic search, no indexing, and no ranking beyond “first N hits up to max chars.”
- **Summarization can fail.** If the summarization LLM call fails, we store a short fallback summary. The virtual tool still works, but section content may be minimal or generic.

---

## Summary

| Caveat | Effect |
|--------|--------|
| Only after compression | No queryable history until context overflow triggers compression. |
| In-memory, no disk | Restart proxy → all stored conversations lost. |
| Single process | Multiple proxy instances don’t share the store. |
| LRU eviction | Old/inactive chats lose history when store is full. |
| One snapshot per conversation | Only latest pre-compression state is queryable. |
| Conversation ID from first user message | Editing first message or collisions can break or mix up history. |
| Result caps | Long results and many search hits are truncated. |
| Section/keyword heuristics | Section may be “not found”; search is literal substring only. |

Use the virtual tool as a best-effort way to pull details from recent compressed context, not as a reliable long-term or multi-instance conversation store.
