"""
콜드패스 모니터.
트레이딩 엔진 상태 감시 및 알림.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


class HealthMonitor:

    def __init__(self, config: dict):
        monitor_cfg = config.get("monitor", {})
        self.heartbeat_interval = monitor_cfg.get("heartbeat_interval_sec", 1)
        self._last_heartbeat = 0
        self._alert_sent = False

    def check_heartbeat(self, engine_heartbeat_ns: int) -> bool:
        """엔진 하트비트 확인. 정상이면 True."""
        now_ns = time.time_ns()
        elapsed_sec = (now_ns - engine_heartbeat_ns) / 1e9

        if elapsed_sec > self.heartbeat_interval * 5:
            if not self._alert_sent:
                log.critical("Engine heartbeat lost — %.1fs since last beat", elapsed_sec)
                self._alert_sent = True
            return False

        self._alert_sent = False
        return True

    def daily_summary(self, daily_pnl: float, trade_count: int) -> dict:
        return {
            "daily_pnl_usd": daily_pnl,
            "trade_count": trade_count,
            "timestamp": time.time(),
        }
