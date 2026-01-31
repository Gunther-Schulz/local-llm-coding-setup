"""Context management: Cursor-style compression + condense large tool responses.

Cursor-style (inferred from proxy log when Cursor Cloud gets 413):
- One [Previous conversation summary] user message with structured sections (Primary Request,
  Key Concepts, Files, Errors, Problem Solving, User messages, Pending Tasks, Current State,
  Optional Next Step) + last N messages verbatim. Trigger: when prompt would exceed backend
  limit (overflow only).
- Long tool responses: we condense to a preview (TOOL_RESPONSE_MAX_VERBATIM /
  TOOL_RESPONSE_PREVIEW_CHARS). Instruction docs (e.g. CLIPPY_MKII.md) can be excluded via
  TOOL_RESPONSE_NO_CONDENSE_PATHS.
- Virtual tool: when we compress we store full messages + summary; proxy injects
  search_compressed_conversation (section / query / message_index) and executes it server-side.
"""
from typing import List, Dict, Any, Optional
import fnmatch
import json
import os
import re
import time
import httpx

# In-memory store for compressed conversations: conversation_id -> { "messages", "summary" }
_MAX_STORED_CONVERSATIONS = 50
_compressed_store: Dict[str, Dict[str, Any]] = {}
_store_order: List[str] = []

from stack.settings import (
    DEBUG,
    BACKEND_URL,
    TOOL_RESPONSE_MAX_VERBATIM,
    TOOL_RESPONSE_PREVIEW_CHARS,
    TOOL_RESPONSE_NO_CONDENSE_PATHS,
    PRESERVE_FIRST_USER_IN_SUMMARY,
    COMPRESSED_STORE_MAX_CONVERSATIONS,
    COMPRESSED_STORE_RESULT_MAX_CHARS,
    COMPRESSED_STORE_SEARCH_TOP_K,
    COMPRESSED_STORE_SEARCH_MAX_CHARS,
)


# Cursor-style structured summary sections (inferred from Cursor Cloud post-413 request shape)
_CURSOR_SUMMARY_SECTIONS = """1.  Primary Request and Intent:
    (What the user asked for and any clarifications.)

2.  Key Technical Concepts:
    (Technologies, patterns, file roles, and terms that matter.)

3.  Files and Code Sections:
    (Important files and what was read/edited, with paths and line references.)

4.  Errors and fixes:
    (What went wrong and how it was fixed.)

5.  Problem Solving:
    (How the task was approached and decisions made.)

6.  All user messages:
    (Short list of each user message or request.)

7.  Pending Tasks:
    (What was left to do or follow-up.)

8.  Current State:
    (Where things stand now.)

9.  Optional Next Step:
    (What the model could do next if the user continues.)"""

# Section param (for virtual tool) -> regex or header to find that block in the summary text
_SECTION_KEYS = [
    "primary_request", "key_concepts", "files_and_code", "errors_and_fixes",
    "problem_solving", "user_messages", "pending_tasks", "current_state", "next_step",
]
_SECTION_HEADERS = [
    "1.  Primary Request and Intent",
    "2.  Key Technical Concepts",
    "3.  Files and Code Sections",
    "4.  Errors and fixes",
    "5.  Problem Solving",
    "6.  All user messages",
    "7.  Pending Tasks",
    "8.  Current State",
    "9.  Optional Next Step",
]
SECTION_PARAM_TO_HEADER = dict(zip(_SECTION_KEYS, _SECTION_HEADERS))

# Virtual tool: name and OpenAI-format definition
VIRTUAL_TOOL_NAME = "search_compressed_conversation"
VIRTUAL_TOOL_DEFINITION: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": VIRTUAL_TOOL_NAME,
        "description": (
            "Query the full pre-compression conversation when the summary above is insufficient. "
            "Provide exactly one of: section (one of primary_request, key_concepts, files_and_code, "
            "errors_and_fixes, problem_solving, user_messages, pending_tasks, current_state, next_step), "
            "query (keyword search over messages), or message_index (0-based index of a single message)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": "One of: primary_request, key_concepts, files_and_code, errors_and_fixes, problem_solving, user_messages, pending_tasks, current_state, next_step",
                    "enum": _SECTION_KEYS,
                },
                "query": {"type": "string", "description": "Keyword search over all messages."},
                "message_index": {"type": "integer", "description": "0-based index of a single message to retrieve."},
            },
        },
    },
}

# Cap size of virtual tool result so we don't blow context
VIRTUAL_TOOL_RESULT_MAX_CHARS = 4000
# For keyword search: max messages to return, and max chars total
SEARCH_TOP_K_MESSAGES = 5
SEARCH_MAX_CHARS = 3500


def _evict_store_if_needed():
    """Evict oldest conversations when over limit. Skip eviction if COMPRESSED_STORE_MAX_CONVERSATIONS <= 0 (unlimited)."""
    max_n = COMPRESSED_STORE_MAX_CONVERSATIONS
    if max_n <= 0:
        return
    while len(_compressed_store) >= max_n and _store_order:
        cid = _store_order.pop(0)
        _compressed_store.pop(cid, None)


def store_compressed(conversation_id: str, messages: List[Dict], summary: str) -> None:
    """Store full uncompressed messages and summary for this conversation. Replaces any previous
    snapshot for this conversation_id so we always keep one full history (multiple compression
    turns during a session overwrite with the latest full pre-compression state)."""
    _evict_store_if_needed()
    if conversation_id in _store_order:
        _store_order.remove(conversation_id)
    _store_order.append(conversation_id)
    _compressed_store[conversation_id] = {"messages": list(messages), "summary": summary}


def get_stored(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Return stored { messages, summary } for a conversation, or None."""
    return _compressed_store.get(conversation_id)


def _extract_section_from_summary(summary: str, section_key: str) -> str:
    """Return the subsection of summary for the given section key, or empty string."""
    header = SECTION_PARAM_TO_HEADER.get(section_key)
    if not header:
        return ""
    # Find this section and the next section (or end)
    idx = summary.find(header)
    if idx < 0:
        return ""
    # Find next "N. " (section start) after this one; allow one or two spaces
    rest = summary[idx + len(header) :]
    match = re.search(r"\n\d+\.\s+\w", rest)
    end_offset = match.start() + 1 if match else len(rest)
    block = summary[idx : idx + len(header) + end_offset]
    return block.strip()


def _message_content_text(msg: Dict) -> str:
    """Extract plain text from a message's content for search."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return str(content)


def execute_virtual_tool(conversation_id: str, arguments: Dict[str, Any]) -> str:
    """
    Execute the virtual tool: section, query, or message_index.
    Returns a string (capped at VIRTUAL_TOOL_RESULT_MAX_CHARS) for the tool result content.
    """
    data = get_stored(conversation_id)
    if not data:
        return "[No compressed conversation stored for this session.]"
    messages = data["messages"]
    summary = data["summary"]

    section = arguments.get("section")
    query = arguments.get("query")
    message_index = arguments.get("message_index")

    provided = sum(1 for x in (section, query, message_index) if x is not None and x != "")
    if provided != 1:
        return "[Use exactly one of: section, query, or message_index.]"

    result = ""
    if section is not None and section != "":
        result = _extract_section_from_summary(summary, section)
        if not result:
            result = f"[Section '{section}' not found in summary.]"
    elif query is not None and str(query).strip():
        q = str(query).strip().lower()
        top_k = _search_top_k()
        max_ch = _search_max_chars()
        hits = []
        total_chars = 0
        for i, msg in enumerate(messages):
            text = _message_content_text(msg)
            if q in text.lower():
                role = msg.get("role", "?")
                preview = text[:800] + "..." if len(text) > 800 else text
                hits.append(f"--- message {i} ({role}) ---\n{preview}")
                total_chars += len(hits[-1])
                if (top_k and len(hits) >= top_k) or (max_ch and total_chars >= max_ch):
                    break
        result = "\n\n".join(hits) if hits else f"[No message contained '{query}'.]"
    elif message_index is not None:
        try:
            idx = int(message_index)
            if 0 <= idx < len(messages):
                msg = messages[idx]
                role = msg.get("role", "?")
                text = _message_content_text(msg)
                cap = _result_max_chars()
                if cap and len(text) > cap:
                    text = text[:cap] + "\n[... truncated]"
                result = f"--- message {idx} ({role}) ---\n{text}"
            else:
                result = f"[message_index {idx} out of range; there are {len(messages)} messages (0-based).]"
        except (TypeError, ValueError):
            result = "[message_index must be an integer.]"
    else:
        result = "[Use exactly one of: section, query, or message_index.]"

    cap = _result_max_chars()
    if cap and len(result) > cap:
        result = result[:cap] + "\n[... truncated]"
    return result


async def summarize_conversation_cursor_style(messages: List[Dict]) -> str:
    """Use the LLM to produce a Cursor-style structured summary (sections like Primary Request, Key Concepts, Files, etc.)."""
    conversation_text = "\n\n".join([
        f"{msg.get('role', 'user')}: {_extract_content_preview(msg, max_chars=800)}"
        for msg in messages
    ])
    summary_prompt = f"""Summarize this conversation history in the following structured format. Use the exact section numbers and titles. Be concise but preserve what the model needs to continue the task.

Sections to produce:
{_CURSOR_SUMMARY_SECTIONS}

Conversation:
{conversation_text}

Your structured summary (use the section numbers and titles above):"""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/v1/chat/completions",
                json={
                    "model": "qwen3-30b-q2",
                    "messages": [{"role": "user", "content": summary_prompt}],
                    "max_tokens": 1500,
                    "temperature": 0.3
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        if DEBUG:
            print(f"[WARNING] Cursor-style summarization failed: {e}, using fallback")
        return f"[Previous conversation with {len(messages)} messages about code work. Primary request and context omitted due to summarization failure.]"


async def compress_cursor_style(
    messages: List[Dict],
    conversation_id: str,
    recent_count: int = 6,
) -> List[Dict]:
    """
    Cursor-style compression: one structured [Previous conversation summary] user message + last N messages.
    Use when prompt would exceed backend limit (overflow). Replaces old sliding-window + short summary.
    """
    # Separate system from conversation
    system_prompt = None
    conversation_messages = messages
    if messages and messages[0].get("role") == "system":
        system_prompt = messages[0]
        conversation_messages = messages[1:]

    if len(conversation_messages) <= recent_count:
        return messages

    split_point = len(conversation_messages) - recent_count
    old_messages = conversation_messages[:split_point]
    recent_messages = conversation_messages[split_point:]

    if DEBUG:
        print(f"[DEBUG] Cursor-style compress: {len(old_messages)} old, {len(recent_messages)} recent (recent_count={recent_count})")

    summary = await summarize_conversation_cursor_style(old_messages)
    if PRESERVE_FIRST_USER_IN_SUMMARY:
        first_user = next((m for m in old_messages if m.get("role") == "user"), None)
        if first_user:
            first_content = _extract_content_preview(first_user, max_chars=2000)
            summary_content = f"[Previous conversation summary]: Summary:\n[Initial user request]:\n{first_content}\n\n{summary}"
        else:
            summary_content = f"[Previous conversation summary]: Summary:\n{summary}"
    else:
        summary_content = f"[Previous conversation summary]: Summary:\n{summary}"

    summary_message = {"role": "user", "content": summary_content}
    condensed_recent = condense_tool_responses_with_context(recent_messages)

    final = []
    if system_prompt:
        final.append(system_prompt)
    final.append(summary_message)
    final.extend(condensed_recent)
    store_compressed(conversation_id, messages, summary_content)
    if DEBUG:
        print(f"[DEBUG] Cursor-style result: {len(final)} messages (1 summary + {len(condensed_recent)} recent); stored for virtual tool")
    return final


def _extract_content_preview(msg: Dict, max_chars: int = 200) -> str:
    """Extract preview of message content."""
    content = msg.get("content", "")
    
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        text = " ".join(text_parts)
    else:
        text = str(content)
    
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def _build_tool_call_id_to_path(messages: List[Dict]) -> Dict[str, str]:
    """Build map tool_call_id -> file path from assistant messages' tool_calls."""
    id_to_path: Dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            tid = tc.get("id")
            if not tid:
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            args = fn.get("arguments")
            if not args:
                continue
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if not isinstance(args, dict):
                continue
            path = args.get("path") or args.get("file_path") or args.get("filename") or args.get("file")
            if path:
                id_to_path[tid] = str(path)
    return id_to_path


def _path_matches_no_condense(path: str, patterns: List[str]) -> bool:
    """True if path (or its basename) matches any fnmatch pattern in patterns."""
    base = os.path.basename(path)
    for p in patterns:
        if fnmatch.fnmatch(path, p) or fnmatch.fnmatch(base, p):
            return True
    return False


def condense_large_tool_response(msg: Dict, skip_condense: bool = False) -> Dict:
    """Condense large tool responses - show preview, keep full text retrievable.
    If skip_condense is True (e.g. instruction doc), return msg unchanged."""
    if msg.get("role") != "tool":
        return msg
    if skip_condense:
        return msg

    content = msg.get("content", "")
    if isinstance(content, list) and len(content) > 0:
        text_content = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
    else:
        text_content = str(content)

    # If tool response is large, condense to preview only
    if len(text_content) > TOOL_RESPONSE_MAX_VERBATIM:
        preview = text_content[:TOOL_RESPONSE_PREVIEW_CHARS] + f"\n\n[... {len(text_content) - TOOL_RESPONSE_PREVIEW_CHARS} more characters omitted ...]"

        condensed = msg.copy()
        if isinstance(content, list):
            condensed["content"] = [{"type": "text", "text": preview}]
        else:
            condensed["content"] = preview

        condensed["_full_content_length"] = len(text_content)
        return condensed

    return msg


def condense_tool_responses_with_context(
    messages: List[Dict],
    no_condense_patterns: Optional[List[str]] = None,
) -> List[Dict]:
    """Condense large tool responses, but skip condensing for paths matching no_condense_patterns
    (e.g. instruction docs like CLIPPY_MKII.md). Patterns use fnmatch (*CLIPPY*.md)."""
    if no_condense_patterns is None:
        no_condense_patterns = TOOL_RESPONSE_NO_CONDENSE_PATHS
    id_to_path = _build_tool_call_id_to_path(messages)
    result = []
    for msg in messages:
        if msg.get("role") != "tool":
            result.append(msg)
            continue
        tid = msg.get("tool_call_id")
        path = id_to_path.get(tid) if tid else None
        skip = bool(path and _path_matches_no_condense(path, no_condense_patterns))
        if DEBUG and skip:
            print(f"[DEBUG] No condense for tool result (path matches): {path}")
        result.append(condense_large_tool_response(msg, skip_condense=skip))
    return result
