"""
브레이크아웃 전략 — 커스텀 전략 예시.

N바 고/저 돌파 시 진입.
BaseStrategy를 상속하여 on_bar()만 구현하면 된다.
"""

from __future__ import annotations

import logging

from src.strategy.base import Bar, BaseStrategy, Direction, Signal

log = logging.getLogger(__name__)


class BreakoutStrategy(BaseStrategy):

    name = "breakout"

    def __init__(self, config: dict):
        breakout_cfg = config.get("breakout", {})
        self.lookback = breakout_cfg.get("lookback_bars", 20)
        self.threshold = breakout_cfg.get("breakout_threshold", 5.0)
        self.tp_points = config.get("take_profit_points", 15)
        self.sl_points = config.get("stop_loss_points", 10)
        self._warmed = False

    def warmup(self, bars: list[Bar]) -> None:
        if len(bars) >= self.lookback:
            self._warmed = True
        log.info("Breakout strategy warmed up (lookback=%d)", self.lookback)

    def on_bar(self, bars: list[Bar]) -> Signal | None:
        if len(bars) < self.lookback + 1:
            return None

        window = bars[-(self.lookback + 1):-1]
        current = bars[-1]

        highest = max(b.high for b in window)
        lowest = min(b.low for b in window)
        channel_range = highest - lowest

        if channel_range < self.threshold:
            return None

        # 상방 돌파
        if current.close > highest:
            breakout_strength = (current.close - highest) / channel_range
            return Signal(
                direction=Direction.LONG,
                confidence=min(breakout_strength, 1.0),
                take_profit=self.tp_points,
                stop_loss=self.sl_points,
                strategy_name=self.name,
                metadata={"highest": highest, "channel_range": channel_range},
            )

        # 하방 돌파
        if current.close < lowest:
            breakout_strength = (lowest - current.close) / channel_range
            return Signal(
                direction=Direction.SHORT,
                confidence=min(breakout_strength, 1.0),
                take_profit=self.tp_points,
                stop_loss=self.sl_points,
                strategy_name=self.name,
                metadata={"lowest": lowest, "channel_range": channel_range},
            )

        return None

    def reset(self) -> None:
        self._warmed = False
