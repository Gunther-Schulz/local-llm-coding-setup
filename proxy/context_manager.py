"""Context management: Cursor-style compression + condense large tool responses.

Cursor-style (inferred from proxy log when Cursor Cloud gets 413):
- One [Previous conversation summary] user message with structured sections (Primary Request,
  Key Concepts, Files, Errors, Problem Solving, User messages, Pending Tasks, Current State,
  Optional Next Step) + last N messages verbatim. Trigger: when prompt would exceed backend
  limit (overflow only).
- Long tool responses: we condense to a preview (TOOL_RESPONSE_MAX_VERBATIM /
  TOOL_RESPONSE_PREVIEW_CHARS). Instruction docs (e.g. CLIPPY_MKII.md) can be excluded via
  TOOL_RESPONSE_NO_CONDENSE_PATHS.
"""
from typing import List, Dict, Any, Optional
import fnmatch
import json
import os
import time
import httpx

from stack.settings import (
    DEBUG,
    BACKEND_URL,
    TOOL_RESPONSE_MAX_VERBATIM,
    TOOL_RESPONSE_PREVIEW_CHARS,
    TOOL_RESPONSE_NO_CONDENSE_PATHS,
    PRESERVE_FIRST_USER_IN_SUMMARY,
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
    if DEBUG:
        print(f"[DEBUG] Cursor-style result: {len(final)} messages (1 summary + {len(condensed_recent)} recent)")
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
