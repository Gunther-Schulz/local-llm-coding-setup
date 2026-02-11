"""
Tool call interceptor for the proxy server.
Injects reminders and prevents read loops via deduplication.
"""

import json
import re
import sys
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TurnState:
    """State tracking for a single turn/interaction."""
    reads: Dict[str, Dict[Tuple[int, int], int]] = field(default_factory=dict)
    """file_path -> {(start, end): count}"""
    read_counts: Dict[str, int] = field(default_factory=dict)
    """file_path -> total read count"""
    overlapping_reads: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    """file_path -> list of ranges read"""


class Interceptor:
    """
    Intercepts tool calls, injects reminders, and prevents loops.
    
    Features:
    - Tool-specific reminder injection
    - Read coalescing (track same file/range reads per turn)
    - Overlapping range detection
    - Turn-based state management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize interceptor with configuration.
        
        Args:
            config: Loaded configuration dict from config/loader.py
        """
        self.config = config
        self.tools_config = config.get("tools", {})
        self.read_coalescing_config = config.get("read_coalescing", {})
        self.overlapping_config = config.get("overlapping_ranges", {})
        
        # Enable/disable features
        self.reminders_enabled = self._get_config_value("tools", True)
        self.read_dedup_enabled = self._get_config_value("read_coalescing", True)
        self.overlapping_detection_enabled = self._get_config_value("overlapping_ranges", True)
        
        # Read limits
        self.max_reads_per_turn = self.read_coalescing_config.get("max_reads_per_turn", 3)
        self.max_overlapping_reads = self.overlapping_config.get("max_overlapping_reads", 2)
        
        # Reminder messages
        self.read_coalescing_reminder = self.read_coalescing_config.get(
            "reminder_message", 
            "WARNING: File/range read multiple times this turn. Avoid loops!"
        )
        self.overlapping_reminder = self.overlapping_config.get(
            "reminder_message",
            "WARNING: Multiple overlapping reads detected. Consider consolidating!"
        )
        
        # Per-turn state
        self.turn_states: Dict[str, TurnState] = {}
        self.max_turns_in_memory = config.get("turn_tracking", {}).get("max_turns_in_memory", 100)
        
        logger.info(f"Interceptor initialized: reminders={self.reminders_enabled}, "
                   f"read_dedup={self.read_dedup_enabled}, overlapping={self.overlapping_detection_enabled}")
    
    def _get_config_value(self, section: str, default: bool) -> bool:
        """Get enabled flag from config section."""
        section_config = self.config.get(section, {})
        return section_config.get("enabled", default)
    
    def reset_turn(self, turn_id: str) -> None:
        """
        Reset state for a new turn.
        
        Args:
            turn_id: Unique identifier for the turn
        """
        had_prev = turn_id in self.turn_states
        self.turn_states[turn_id] = TurnState()
        logger.debug("reset_turn: turn_id=%s had_prev=%s turns_in_memory=%d", turn_id, had_prev, len(self.turn_states))
        
        if len(self.turn_states) > self.max_turns_in_memory:
            oldest_keys = list(self.turn_states.keys())[:-self.max_turns_in_memory]
            for key in oldest_keys:
                del self.turn_states[key]
            logger.debug("trimmed turn_states: removed %d oldest, kept %d", len(oldest_keys), self.max_turns_in_memory)
    
    def intercept_call(self, tool_call: Dict[str, Any], turn_id: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Intercept a tool call and potentially inject a reminder.
        
        Args:
            tool_call: Tool call JSON from Cursor IDE
            turn_id: Optional turn identifier for state tracking
            
        Returns:
            Tuple of (modified_tool_call, reminder_message or None)
        """
        # Support both flat (tool/name) and OpenAI-style (function.name) tool call format
        fn = tool_call.get("function") or {}
        tool_name = (
            tool_call.get("tool")
            or tool_call.get("name")
            or (fn.get("name") if isinstance(fn, dict) else "")
            or ""
        )
        logger.debug("intercept_call: tool=%s turn_id=%s", tool_name, turn_id)
        
        reminder = None
        if self.reminders_enabled:
            reminder = self._get_reminder_for_tool(tool_name)
            logger.debug("reminder lookup: tool=%s enabled=%s has_message=%s", tool_name, True, reminder is not None)
        
        if tool_name == "Read":
            read_reminder = self._track_read(tool_call, turn_id)
            if read_reminder:
                if reminder:
                    reminder = reminder + "\n\n" + read_reminder
                else:
                    reminder = read_reminder
        
        if reminder:
            logger.debug("reminder for tool=%s len=%d (delivered via message)", tool_name, len(reminder))
        
        return tool_call, reminder
    
    def _get_reminder_for_tool(self, tool_name: str) -> Optional[str]:
        """Get reminder message for a tool if configured."""
        tool_config = self.tools_config.get(tool_name, {})
        
        if not tool_config.get("enabled", True):
            return None
        
        return tool_config.get("message", None)
    
    def _track_read(self, tool_call: Dict[str, Any], turn_id: Optional[str]) -> Optional[str]:
        """
        Track a Read tool call and detect loops.
        
        Args:
            tool_call: Read tool call
            turn_id: Current turn identifier
            
        Returns:
            Warning message if loop detected, None otherwise
        """
        if not self.read_dedup_enabled:
            logger.debug("read tracking disabled, skipping")
            return None
        
        if turn_id is None:
            logger.debug("read tracking without turn_id, skipping (reminders still applied)")
            return None
        
        file_path, range_tuple = self._parse_read_params(tool_call)
        logger.debug("parse_read_params: file_path=%s range=%s", file_path, range_tuple)
        
        if not file_path:
            logger.warning("could not parse file path from Read call: %s", list((tool_call.get("params") or tool_call.get("arguments") or {}).keys()))
            return None
        
        if turn_id not in self.turn_states:
            self.turn_states[turn_id] = TurnState()
        state = self.turn_states[turn_id]
        
        if file_path not in state.reads:
            state.reads[file_path] = {}
        if range_tuple not in state.reads[file_path]:
            state.reads[file_path][range_tuple] = 0
        state.reads[file_path][range_tuple] += 1
        if file_path not in state.read_counts:
            state.read_counts[file_path] = 0
        state.read_counts[file_path] += 1
        if file_path not in state.overlapping_reads:
            state.overlapping_reads[file_path] = []
        state.overlapping_reads[file_path].append(range_tuple)
        
        read_count = state.reads[file_path][range_tuple]
        total_file_reads = state.read_counts[file_path]
        logger.debug("read state: file=%s range=%s same_range_count=%d total_file_reads=%d", file_path, range_tuple, read_count, total_file_reads)
        
        if read_count >= self.max_reads_per_turn:
            msg = self.read_coalescing_reminder.format(count=read_count)
            logger.info("loop detected: file=%s range=%s count=%d >= max=%d", file_path, range_tuple, read_count, self.max_reads_per_turn)
            return msg
        
        overlapping_count = self._count_overlapping_reads(state.overlapping_reads.get(file_path, []))
        logger.debug("overlapping_reads: file=%s count=%d max_allowed=%d", file_path, overlapping_count, self.max_overlapping_reads)
        if overlapping_count >= self.max_overlapping_reads:
            logger.info("overlapping reads: file=%s count=%d >= max=%d", file_path, overlapping_count, self.max_overlapping_reads)
            return self.overlapping_reminder
        
        return None
    
    def _parse_read_params(self, tool_call: Dict[str, Any]) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        """
        Extract file path and range from Read tool call.
        
        Args:
            tool_call: Read tool call JSON
            
        Returns:
            Tuple of (file_path, (start_line, end_line)) or (None, None) if parsing fails
        """
        params = tool_call.get("params") or tool_call.get("arguments")
        if params is None:
            fn = tool_call.get("function") or {}
            params = fn.get("arguments", {}) if isinstance(fn, dict) else {}
        if isinstance(params, str):
            try:
                params = json.loads(params) if params else {}
            except (ValueError, TypeError):
                params = {}
        if not isinstance(params, dict):
            params = {}
        
        # Try different parameter names
        file_path = params.get("file", params.get("path", params.get("filePath", None)))
        
        # Range can be specified in various ways
        range_params = params.get("range", params.get("selection", params.get("lines", None)))
        
        start_line = None
        end_line = None
        
        if range_params:
            if isinstance(range_params, dict):
                start_line = range_params.get("start", range_params.get("startLine", None))
                end_line = range_params.get("end", range_params.get("endLine", None))
            elif isinstance(range_params, (list, tuple)) and len(range_params) >= 2:
                start_line = range_params[0]
                end_line = range_params[1]
        
        if start_line is None:
            start_line = 0
        if end_line is None:
            # No range specified - read full file (use sentinel int, not inf - int(inf) raises)
            end_line = sys.maxsize
        
        try:
            range_tuple = (int(start_line), int(end_line))
        except (TypeError, ValueError, OverflowError):
            logger.debug("parse_read_params: invalid range start=%s end=%s", start_line, end_line)
            range_tuple = (0, 0)
        return file_path, range_tuple
    
    def _count_overlapping_reads(self, ranges: List[Tuple[int, int]]) -> int:
        """
        Count how many ranges overlap with each other.
        
        Args:
            ranges: List of (start, end) tuples
            
        Returns:
            Count of ranges that have overlap with at least one other range
        """
        if len(ranges) < 2:
            return 0
        
        overlapping_count = 0
        
        for i, (start1, end1) in enumerate(ranges):
            for j, (start2, end2) in enumerate(ranges):
                if i >= j:
                    continue
                # Check if ranges overlap
                if start1 <= end2 and start2 <= end1:
                    overlapping_count += 1
                    break
        
        return overlapping_count