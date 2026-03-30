"""
시장 상태 분류기 (Market Regime Detector).

시장 상태를 실시간으로 판별하여 전략 매니저에게 제공.
전략 매니저는 이 정보를 기반으로 전략을 자동 전환한다.

시장 상태 (09:35~10:30 ET 개장 직후 구간):
  TRENDING    — 명확한 방향성, 모멘텀 전략 유리
  RANGING     — 박스권, 브레이크아웃 전략 유리
  VOLATILE    — 급변동, 보수적 진입 (확신도 기준 강화)

  주의: QUIET 는 거래 시간대(개장 직후) 특성상 거의 발생하지 않으나
        지표 발표 전 잠시 나타날 수 있으므로 유지.
"""

from __future__ import annotations

import logging
from enum import Enum, auto

import numpy as np

from src.strategy.base import Bar

log = logging.getLogger(__name__)


class MarketRegime(Enum):
    TRENDING = auto()
    RANGING = auto()
    VOLATILE = auto()
    QUIET = auto()


class RegimeDetector:
    """
    최근 N바의 통계를 기반으로 시장 상태를 판별.

    판별 기준:
      - ADX 근사 (방향성 강도) → TRENDING vs RANGING
      - 바 소요시간 (duration) → QUIET 판별
      - 가격 변동폭 (ATR 대비) → VOLATILE 판별
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self._current = MarketRegime.RANGING
        self._prev = MarketRegime.RANGING

        # 임계값 (백테스트로 튜닝)
        self.adx_trend_threshold = 25.0       # 이상이면 TRENDING
        self.atr_volatile_multiplier = 2.0    # 평균 ATR의 2배 이상이면 VOLATILE
        self.quiet_duration_ms = 60_000       # 바 1개 60초 이상이면 QUIET

    def update(self, bars: list[Bar]) -> MarketRegime:
        """최근 바들을 분석하여 시장 상태를 갱신."""
        if len(bars) < self.lookback:
            return self._current

        recent = bars[-self.lookback:]
        self._prev = self._current

        # 1) QUIET 체크 — 최근 바들의 평균 소요시간
        avg_duration = np.mean([b.duration_ms for b in recent])
        if avg_duration > self.quiet_duration_ms:
            self._current = MarketRegime.QUIET
            return self._current

        # 2) VOLATILE 체크 — ATR 급증
        ranges = [b.range for b in recent]
        current_atr = np.mean(ranges[-5:])     # 최근 5바 ATR
        baseline_atr = np.mean(ranges)          # 전체 평균 ATR
        if baseline_atr > 0 and current_atr > baseline_atr * self.atr_volatile_multiplier:
            self._current = MarketRegime.VOLATILE
            return self._current

        # 3) TRENDING vs RANGING — 방향성 판별
        closes = [b.close for b in recent]
        directional_move = abs(closes[-1] - closes[0])
        total_move = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))

        if total_move == 0:
            efficiency = 0.0
        else:
            efficiency = directional_move / total_move  # 0~1, 높을수록 추세

        # efficiency > 0.4 이면 추세로 판단 (ADX 25 근사)
        if efficiency > 0.4:
            self._current = MarketRegime.TRENDING
        else:
            self._current = MarketRegime.RANGING

        return self._current

    @property
    def regime(self) -> MarketRegime:
        return self._current

    @property
    def changed(self) -> bool:
        return self._current != self._prev
