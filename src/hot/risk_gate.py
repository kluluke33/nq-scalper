"""
리스크 게이트. 인메모리 동기 체크. DB/파일 조회 없음.
모든 주문은 이 게이트를 통과해야 실행된다.

시간 관련 판단은 SessionManager가 담당.
RiskGate는 포지션/손실/연속손실만 판단.
"""

from __future__ import annotations

import logging
import time

from src.strategy.base import Signal

log = logging.getLogger(__name__)


class RiskGate:
    __slots__ = (
        "max_contracts", "max_daily_loss", "max_consecutive_losses",
        "cooldown_sec",
        "daily_pnl", "consecutive_losses", "open_qty",
        "last_loss_time", "_trade_count",
    )

    def __init__(self, config: dict):
        risk = config.get("risk", config)
        self.max_contracts = risk.get("max_contracts", 2)
        self.max_daily_loss = -abs(risk.get("max_daily_loss_usd", 200))
        self.max_consecutive_losses = risk.get("max_consecutive_losses", 3)
        self.cooldown_sec = risk.get("cooldown_minutes", 10) * 60  # 55분 윈도우이므로 10분

        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.open_qty = 0
        self.last_loss_time = 0.0
        self._trade_count = 0

    def allow(self, signal: Signal) -> bool:
        """포지션/손실 기준만 체크. 시간 판단은 SessionManager가 담당."""
        if self.daily_pnl <= self.max_daily_loss:
            log.warning("RISK BLOCK: daily loss limit (%.2f)", self.daily_pnl)
            return False

        if self.consecutive_losses >= self.max_consecutive_losses:
            elapsed = time.time() - self.last_loss_time
            if elapsed < self.cooldown_sec:
                remaining = int(self.cooldown_sec - elapsed)
                log.warning("RISK BLOCK: cooldown (%ds remaining)", remaining)
                return False
            self.consecutive_losses = 0

        if self.open_qty >= self.max_contracts:
            return False

        return True

    def on_fill(self, qty: int) -> None:
        self.open_qty += qty
        self._trade_count += 1

    def on_close(self, pnl: float) -> None:
        self.daily_pnl += pnl
        self.open_qty = max(0, self.open_qty - 1)

        if pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_time = time.time()
        else:
            self.consecutive_losses = 0

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.open_qty = 0
        self._trade_count = 0

    @property
    def stats(self) -> dict:
        return {
            "daily_pnl": self.daily_pnl,
            "trade_count": self._trade_count,
            "consecutive_losses": self.consecutive_losses,
            "open_qty": self.open_qty,
        }
