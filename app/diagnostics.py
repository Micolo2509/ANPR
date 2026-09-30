from __future__ import annotations

import logging
import os
from pathlib import Path


def process_rss_mib() -> float | None:
    try:
        statm = Path("/proc/self/statm").read_text().split()
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(statm[1]) * page_size / (1024 * 1024)
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def log_phase(logger: logging.Logger, phase: str, **fields: object) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    rss = process_rss_mib()
    if rss is not None:
        details = f"{details} rss_mib={rss:.1f}" if details else f"rss_mib={rss:.1f}"
    logger.info("%s%s", phase, f" {details}" if details else "")
