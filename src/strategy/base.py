"""
전략 베이스 클래스.
모든 커스텀 전략은 이 클래스를 상속받아 구현한다.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Direction(IntEnum):
    LONG = 1
    SHORT = -1


@dataclass(slots=True)
class Signal:
    direction: Direction
    confidence: float          # 0.0 ~ 1.0
    take_profit: float         # 목표 수익 (포인트)
    stop_loss: float           # 손절 (포인트)
    strategy_name: str = ""
    timestamp_ns: int = 0      # 신호 생성 시각 (나노초)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp_ns == 0:
            self.timestamp_ns = time.time_ns()


@dataclass(slots=True)
class Bar:
    open: float
    high: float
    low: float
    close: float
    tick_count: int
    buy_ticks: int
    sell_ticks: int
    duration_ms: float         # 바 완성까지 소요된 시간 (ms)
    timestamp_ns: int = 0

    @property
    def delta(self) -> int:
        return self.buy_ticks - self.sell_ticks

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def close_position(self) -> float:
        """바 내 종가의 상대 위치 (0.0=저점, 1.0=고점)"""
        r = self.range
        if r == 0:
            return 0.5
        return (self.close - self.low) / r


class BaseStrategy(ABC):
    """
    모든 전략이 구현해야 할 인터페이스.

    사용법:
        class MyStrategy(BaseStrategy):
            name = "my_strategy"

            def warmup(self, bars):
                # 과거 데이터로 내부 상태 초기화
                ...

            def on_bar(self, bars):
                # 바 완성 시 호출, Signal 또는 None 반환
                ...
    """

    name: str = "base"

    @abstractmethod
    def warmup(self, bars: list[Bar]) -> None:
        """
        과거 데이터로 내부 상태를 초기화한다.
        엔진 시작 시 1회 호출.
        """

    @abstractmethod
    def on_bar(self, bars: list[Bar]) -> Signal | None:
        """
        새 바가 완성되면 호출.
        진입 신호가 있으면 Signal, 없으면 None 반환.
        """

    def on_tick_update(self, bars: list[Bar], buffer_ratio: float) -> None:
        """
        틱이 들어올 때마다 호출 (선택적 구현).
        buffer_ratio: 현재 바의 틱 채움 비율 (0.0 ~ 1.0).
        사전 준비(pre-warming)가 필요한 전략만 오버라이드.
        """

    def reset(self) -> None:
        """전략 상태 초기화. 일일 리셋 등에 사용."""
