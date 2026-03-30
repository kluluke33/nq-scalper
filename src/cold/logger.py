"""
콜드패스 로거.
핫패스의 EventBuffer를 주기적으로 읽어 DB/파일에 기록.
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class TradeLogger:

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d")
        self._trade_file = self.log_dir / f"trades_{date_str}.csv"
        self._init_csv()

    def _init_csv(self) -> None:
        if not self._trade_file.exists():
            with open(self._trade_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "type", "direction", "price",
                    "pnl_points", "pnl_usd", "confidence",
                    "strategy", "action",
                ])

    def log_event(self, event: dict[str, Any]) -> None:
        row = [
            datetime.now().isoformat(),
            event.get("type", ""),
            event.get("direction", ""),
            event.get("price", ""),
            event.get("pnl_points", ""),
            event.get("pnl_usd", ""),
            event.get("confidence", ""),
            event.get("strategy", ""),
            event.get("action", ""),
        ]
        with open(self._trade_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
