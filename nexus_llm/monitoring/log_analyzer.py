"""Log analysis for Nexus-LLM monitoring."""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LogEntry:
    """A parsed log line."""
    timestamp: Optional[datetime]
    level: str
    message: str
    raw: str
    line_number: int


@dataclass
class LogReport:
    """Summary report from log analysis."""
    file_path: str
    total_lines: int
    parsed_lines: int
    level_counts: Dict[str, int]
    error_entries: List[LogEntry]
    patterns: Dict[str, int]
    time_range: Optional[Tuple[datetime, datetime]]
    statistics: Dict[str, Any]


# Common log line patterns
_LOG_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*"
    r"(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)?\s*"
    r"(?P<message>.*)"
)

_LEVEL_ALIASES = {
    "WARN": "WARNING",
    "FATAL": "CRITICAL",
}


class LogAnalyzer:
    """Analyzes log files for errors, patterns, and statistics.

    Supports common log formats with timestamp and level prefixes.
    Falls back gracefully for non-standard formats.
    """

    # Default patterns to search for
    DEFAULT_PATTERNS: List[str] = [
        r"Exception",
        r"Traceback",
        r"Error",
        r"timeout",
        r"OOM",
        r"CUDA",
        r"connection refused",
        r"permission denied",
    ]

    def __init__(self, custom_patterns: Optional[List[str]] = None) -> None:
        self._custom_patterns = custom_patterns or []
        self._entries: List[LogEntry] = []
        self._analyzed: bool = False

    def analyze_log_file(self, path: str) -> LogReport:
        """Parse and analyze a log file.

        Args:
            path: Filesystem path to the log file.

        Returns:
            ``LogReport`` with analysis results.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Log file not found: {path}")

        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        self._entries = self._parse_lines(lines, str(file_path))
        self._analyzed = True

        return self._build_report(str(file_path), len(lines))

    def find_errors(self, since: Optional[datetime] = None) -> List[LogEntry]:
        """Find error-level log entries, optionally after a given time.

        Args:
            since: If provided, only return entries after this datetime.

        Returns:
            List of ``LogEntry`` objects with ERROR/CRITICAL level.
        """
        if not self._analyzed:
            return []

        error_levels = {"ERROR", "CRITICAL"}
        results: List[LogEntry] = []

        for entry in self._entries:
            if entry.level not in error_levels:
                continue
            if since is not None and entry.timestamp is not None:
                if entry.timestamp < since:
                    continue
            results.append(entry)

        return results

    def extract_patterns(self) -> Dict[str, int]:
        """Extract and count occurrences of known patterns in log messages.

        Returns:
            Dict mapping pattern strings to occurrence counts.
        """
        if not self._analyzed:
            return {}

        all_patterns = self.DEFAULT_PATTERNS + self._custom_patterns
        pattern_counts: Dict[str, int] = {}

        for pattern in all_patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error:
                continue
            count = sum(1 for e in self._entries if compiled.search(e.message))
            if count > 0:
                pattern_counts[pattern] = count

        return pattern_counts

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics from analyzed log data.

        Returns:
            Dict with keys: ``total_entries``, ``level_distribution``,
            ``error_rate``, ``entries_per_hour``, ``top_error_messages``.
        """
        if not self._analyzed:
            return {}

        level_counts = Counter(e.level for e in self._entries)
        total = len(self._entries)
        error_count = level_counts.get("ERROR", 0) + level_counts.get("CRITICAL", 0)

        # Entries per hour
        hourly: Dict[str, int] = defaultdict(int)
        for entry in self._entries:
            if entry.timestamp is not None:
                hour_key = entry.timestamp.strftime("%Y-%m-%d %H:00")
                hourly[hour_key] += 1

        # Top error messages
        error_msgs = Counter(
            e.message[:200] for e in self._entries
            if e.level in {"ERROR", "CRITICAL"}
        )

        return {
            "total_entries": total,
            "level_distribution": dict(level_counts),
            "error_rate": round(error_count / total, 4) if total > 0 else 0.0,
            "entries_per_hour": dict(hourly),
            "top_error_messages": dict(error_msgs.most_common(10)),
        }

    # -- Private helpers ------------------------------------------------------

    @staticmethod
    def _parse_lines(lines: List[str], source: str) -> List[LogEntry]:
        """Parse raw log lines into structured ``LogEntry`` objects."""
        entries: List[LogEntry] = []

        for idx, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = _LOG_PATTERN.match(line)
            if match:
                ts_str = match.group("timestamp")
                level = match.group("level") or "INFO"
                message = match.group("message").strip()

                # Normalize level
                level = _LEVEL_ALIASES.get(level, level)

                # Parse timestamp
                timestamp = None
                if ts_str:
                    for fmt in (
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%d %H:%M:%S.%f",
                        "%Y/%m/%d %H:%M:%S",
                        "%Y/%m/%d %H:%M:%S.%f",
                        "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S.%f",
                    ):
                        try:
                            timestamp = datetime.strptime(ts_str, fmt)
                            break
                        except ValueError:
                            continue

                entries.append(LogEntry(
                    timestamp=timestamp,
                    level=level,
                    message=message,
                    raw=raw_line,
                    line_number=idx,
                ))
            else:
                # Non-standard line — treat as INFO
                entries.append(LogEntry(
                    timestamp=None,
                    level="INFO",
                    message=line,
                    raw=raw_line,
                    line_number=idx,
                ))

        return entries

    def _build_report(self, file_path: str, total_lines: int) -> LogReport:
        """Build a ``LogReport`` from parsed entries."""
        parsed = len(self._entries)
        level_counts = dict(Counter(e.level for e in self._entries))

        error_entries = [
            e for e in self._entries if e.level in {"ERROR", "CRITICAL"}
        ]

        timestamps = [e.timestamp for e in self._entries if e.timestamp is not None]
        time_range = None
        if timestamps:
            time_range = (min(timestamps), max(timestamps))

        return LogReport(
            file_path=file_path,
            total_lines=total_lines,
            parsed_lines=parsed,
            level_counts=level_counts,
            error_entries=error_entries,
            patterns=self.extract_patterns(),
            time_range=time_range,
            statistics=self.get_statistics(),
        )
