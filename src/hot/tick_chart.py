"""
틱 차트 바 생성기.

틱 수 결정 방식:
  - 고정 모드: 항상 동일한 N틱으로 바 생성
  - 적응형 모드: 최근 틱 유입 속도(TPS)를 측정하여 틱 수를 자동 조절
    → 활발한 시장: 틱 수 증가 (노이즈 필터링)
    → 조용한 시장: 틱 수 감소 (민감도 유지)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from src.strategy.base import Bar


@dataclass(slots=True)
class Tick:
    price: float
    size: int
    side: int          # 1=buy, -1=sell, 0=unknown
    timestamp_ns: int


class AdaptiveTickSizer:
    """
    틱 유입 속도(TPS)를 측정하여 적절한 틱 수를 계산.

    원리:
      - TPS가 높으면 (시장 활발) → 틱 수를 늘려 노이즈를 줄임
      - TPS가 낮으면 (시장 한산) → 틱 수를 줄여 반응성 유지
      - 결과적으로 바 1개의 시간 길이가 대략 일정해짐 (target_bar_duration)

    설정 예시:
      target_bar_duration_sec: 30   → 바 1개가 약 30초 분량이 되도록 조절
      min_ticks: 100                → 아무리 한산해도 최소 100틱
      max_ticks: 2000               → 아무리 활발해도 최대 2000틱
      smoothing: 0.1                → TPS 변화를 부드럽게 반영 (0~1, 낮을수록 안정)
    """

    def __init__(
        self,
        target_bar_duration_sec: float = 30.0,
        min_ticks: int = 100,
        max_ticks: int = 2000,
        default_ticks: int = 500,
        smoothing: float = 0.1,
    ):
        self.target_duration = target_bar_duration_sec
        self.min_ticks = min_ticks
        self.max_ticks = max_ticks
        self.smoothing = smoothing

        self._current_ticks = default_ticks
        self._smoothed_tps = 0.0       # 평활화된 초당 틱 수
        self._window_ticks = 0         # 현재 측정 윈도우의 틱 카운트
        self._window_start_ns = 0      # 현재 측정 윈도우 시작 시각
        self._initialized = False

    @property
    def current_tick_count(self) -> int:
        return self._current_ticks

    def on_tick(self, timestamp_ns: int) -> None:
        """매 틱마다 호출. TPS를 측정하고 틱 수를 갱신."""
        if not self._initialized:
            self._window_start_ns = timestamp_ns
            self._initialized = True

        self._window_ticks += 1

        # 5초마다 TPS 재계산
        elapsed_ns = timestamp_ns - self._window_start_ns
        if elapsed_ns >= 5_000_000_000:  # 5초
            elapsed_sec = elapsed_ns / 1e9
            measured_tps = self._window_ticks / elapsed_sec

            # 지수이동평균으로 TPS 평활화
            if self._smoothed_tps == 0:
                self._smoothed_tps = measured_tps
            else:
                self._smoothed_tps = (
                    self.smoothing * measured_tps
                    + (1 - self.smoothing) * self._smoothed_tps
                )

            # 목표: target_duration초 동안의 틱 수
            ideal_ticks = int(self._smoothed_tps * self.target_duration)
            self._current_ticks = max(self.min_ticks, min(self.max_ticks, ideal_ticks))

            # 윈도우 리셋
            self._window_ticks = 0
            self._window_start_ns = timestamp_ns

    def get_stats(self) -> dict:
        return {
            "smoothed_tps": round(self._smoothed_tps, 1),
            "current_tick_count": self._current_ticks,
        }


class TickChart:

    def __init__(
        self,
        tick_count: int = 500,
        max_bars: int = 2000,
        adaptive: bool = False,
        adaptive_config: dict | None = None,
    ):
        self.tick_count = tick_count
        self.max_bars = max_bars
        self.bars: deque[Bar] = deque(maxlen=max_bars)
        self._buffer: list[Tick] = []
        self._bar_start_ns: int = 0

        # 적응형 틱 사이저
        self._adaptive = adaptive
        self._sizer: AdaptiveTickSizer | None = None
        if adaptive:
            cfg = adaptive_config or {}
            self._sizer = AdaptiveTickSizer(
                target_bar_duration_sec=cfg.get("target_bar_duration_sec", 30.0),
                min_ticks=cfg.get("min_ticks", 100),
                max_ticks=cfg.get("max_ticks", 2000),
                default_ticks=tick_count,
                smoothing=cfg.get("smoothing", 0.1),
            )

    @property
    def effective_tick_count(self) -> int:
        """현재 적용 중인 틱 수."""
        if self._sizer:
            return self._sizer.current_tick_count
        return self.tick_count

    @property
    def buffer_ratio(self) -> float:
        """현재 바의 틱 채움 비율 (0.0 ~ 1.0)."""
        return len(self._buffer) / self.effective_tick_count

    @property
    def current_partial_bar(self) -> Bar | None:
        """형성 중인 미완성 바 (예비 추론용)."""
        if not self._buffer:
            return None
        return self._build_bar(self._buffer)

    def on_tick(self, tick: Tick) -> Bar | None:
        """
        틱 1개 추가. 바가 완성되면 Bar 반환, 아니면 None.
        적응형 모드에서는 TPS에 따라 바 크기가 자동 조절됨.
        """
        # 적응형 틱 수 업데이트
        if self._sizer:
            self._sizer.on_tick(tick.timestamp_ns)

        if not self._buffer:
            self._bar_start_ns = tick.timestamp_ns

        self._buffer.append(tick)

        if len(self._buffer) >= self.effective_tick_count:
            bar = self._build_bar(self._buffer)
            self.bars.append(bar)
            self._buffer = []
            return bar

        return None

    def _build_bar(self, ticks: list[Tick]) -> Bar:
        prices = [t.price for t in ticks]
        buy_ticks = sum(1 for t in ticks if t.side == 1)
        sell_ticks = sum(1 for t in ticks if t.side == -1)

        now = time.time_ns()
        duration_ms = (now - self._bar_start_ns) / 1_000_000

        return Bar(
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            tick_count=len(ticks),
            buy_ticks=buy_ticks,
            sell_ticks=sell_ticks,
            duration_ms=duration_ms,
            timestamp_ns=now,
        )

    def bars_as_list(self) -> list[Bar]:
        return list(self.bars)
