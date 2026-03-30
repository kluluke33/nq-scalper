"""
AI 기반 진입 전략.

핵심 설계:
  - 모델은 시작 시 메모리에 상주
  - 피처는 점진적으로 업데이트 (매 바 O(1))
  - 바 완성 임박 시 예비 추론 실행 → 완성 시 즉시 신호
"""

from __future__ import annotations

import logging
import time

import numpy as np

from src.strategy.base import Bar, BaseStrategy, Direction, Signal

log = logging.getLogger(__name__)


class IncrementalFeatures:
    """
    점진적 피처 계산기.

    피처 30개:
      [가격 구조]
        f[0]  close               현재 종가
        f[1]  change              직전 바 대비 가격 변화
        f[2]  ema_fast            EMA(8) — 단기 추세
        f[3]  ema_slow            EMA(21) — 중기 추세
        f[4]  macd                EMA(8) - EMA(21) — 추세 강도
        f[5]  rsi                 RSI(14) — 과매수/과매도
        f[6]  bar_range           바의 고가-저가 — 변동성
        f[7]  bar_body            |종가-시가| — 실체 크기
        f[8]  close_position      바 내 종가 위치 (0~1)

      [주문 흐름]
        f[9]  delta               매수틱-매도틱 — 체결 압력
        f[10] buy_ratio           매수틱 비율 — 체결 강도
        f[11] cum_delta           누적 델타 — 일중 매수/매도 흐름
        f[12] duration_ms         바 완성 소요시간 — 활동량

      [일중 컨텍스트]
        f[13] day_position        일중 고/저 대비 현재 위치
        f[14] day_high            당일 고가
        f[15] day_low             당일 저가
        f[16] dist_from_ema_fast  가격-EMA(8) 괴리
        f[17] dist_from_ema_slow  가격-EMA(21) 괴리

      [시퀀스 대체 — 직전 바 요약]
        f[18] momentum_3bar       직전 3바 종가 변화 합계 — 단기 모멘텀
        f[19] momentum_5bar       직전 5바 종가 변화 합계 — 중기 모멘텀
        f[20] delta_sum_3bar      직전 3바 델타 합계 — 매수/매도 추세
        f[21] delta_sum_5bar      직전 5바 델타 합계
        f[22] bullish_ratio_3bar  직전 3바 중 양봉 비율 — 방향 일관성
        f[23] range_ratio         직전 5바 평균 range 대비 현재 range — 변동성 변화

      [개장 직후 특화]
        f[24] gap                 전일 종가 대비 갭 (포인트)
        f[25] gap_fill_ratio      갭 채움 비율 (0~1)
        f[26] minutes_since_open  개장 후 경과 시간 (분)
        f[27] opening_range_pos   초기 5분 레인지 대비 현재 위치
        f[28] session_direction   개장 후 누적 방향성 (효율 비율)
        f[29] session_momentum    개장 후 순 변화 (포인트)
    """

    FEATURE_COUNT = 30

    def __init__(self):
        self.features = np.zeros(self.FEATURE_COUNT, dtype=np.float64)

        # 내부 상태
        self._ema_fast = 0.0
        self._ema_slow = 0.0
        self._rsi_gain_avg = 0.0
        self._rsi_loss_avg = 0.0
        self._prev_close = 0.0
        self._cum_delta = 0
        self._bar_count = 0
        self._day_high = -np.inf
        self._day_low = np.inf

        # 시퀀스 대체용 링버퍼 (최근 5바)
        self._recent_closes = np.zeros(5, dtype=np.float64)
        self._recent_deltas = np.zeros(5, dtype=np.float64)
        self._recent_ranges = np.zeros(5, dtype=np.float64)
        self._recent_bodies = np.zeros(5, dtype=np.float64)  # 양봉/음봉 판별용

        # 개장 직후 특화
        self._prev_day_close = 0.0        # 전일 종가
        self._session_first_close = 0.0   # 개장 첫 바 종가
        self._opening_range_high = 0.0    # 초기 5분 고가
        self._opening_range_low = np.inf  # 초기 5분 저가
        self._opening_range_set = False   # OR 확정 여부
        self._session_start_ns = 0        # 세션 시작 시각
        self._session_bar_count = 0       # 세션 내 바 수

    def set_prev_day_close(self, price: float) -> None:
        """전일 종가 설정. 엔진 시작 시 1회 호출."""
        self._prev_day_close = price

    def set_session_start(self, timestamp_ns: int) -> None:
        """세션 시작 시각 설정."""
        self._session_start_ns = timestamp_ns

    def warmup(self, bars: list[Bar]) -> None:
        for bar in bars:
            self._update_internals(bar)

    def update(self, bar: Bar) -> np.ndarray:
        self._update_internals(bar)
        return self.features

    def preview(self, bar: Bar) -> np.ndarray:
        """예비 추론용. 내부 상태를 변경하지 않고 피처를 계산."""
        preview = self.features.copy()
        change = bar.close - self._prev_close if self._prev_close else 0.0
        preview[0] = bar.close
        preview[1] = change
        preview[6] = bar.range
        preview[7] = bar.body
        preview[8] = bar.close_position
        preview[9] = bar.delta
        preview[10] = bar.buy_ticks / bar.tick_count if bar.tick_count > 0 else 0.5
        return preview

    def _update_internals(self, bar: Bar) -> None:
        self._bar_count += 1
        self._session_bar_count += 1
        close = bar.close
        change = close - self._prev_close if self._prev_close else 0.0

        # ── EMA ──
        alpha_fast = 2.0 / (8 + 1)
        alpha_slow = 2.0 / (21 + 1)
        if self._bar_count == 1:
            self._ema_fast = close
            self._ema_slow = close
        else:
            self._ema_fast = alpha_fast * close + (1 - alpha_fast) * self._ema_fast
            self._ema_slow = alpha_slow * close + (1 - alpha_slow) * self._ema_slow

        # ── RSI (Wilder smoothing) ──
        gain = max(change, 0)
        loss = max(-change, 0)
        if self._bar_count <= 14:
            self._rsi_gain_avg += gain / 14
            self._rsi_loss_avg += loss / 14
        else:
            self._rsi_gain_avg = (self._rsi_gain_avg * 13 + gain) / 14
            self._rsi_loss_avg = (self._rsi_loss_avg * 13 + loss) / 14

        rsi = 100.0
        if self._rsi_loss_avg > 0:
            rs = self._rsi_gain_avg / self._rsi_loss_avg
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # ── 누적 델타 ──
        self._cum_delta += bar.delta

        # ── 일중 고/저 ──
        self._day_high = max(self._day_high, bar.high)
        self._day_low = min(self._day_low, bar.low)
        day_range = self._day_high - self._day_low
        day_position = (close - self._day_low) / day_range if day_range > 0 else 0.5

        # ── 링버퍼 업데이트 (최근 5바) ──
        idx = (self._bar_count - 1) % 5
        self._recent_closes[idx] = close
        self._recent_deltas[idx] = bar.delta
        self._recent_ranges[idx] = bar.range
        self._recent_bodies[idx] = 1.0 if bar.close >= bar.open else 0.0

        # ── 시퀀스 대체 피처 계산 ──
        n = min(self._bar_count, 5)

        # 직전 3바 모멘텀
        if n >= 3:
            closes_3 = [self._recent_closes[(self._bar_count - 1 - i) % 5] for i in range(3)]
            momentum_3 = closes_3[0] - closes_3[2]
            delta_sum_3 = sum(self._recent_deltas[(self._bar_count - 1 - i) % 5] for i in range(3))
            bullish_3 = sum(self._recent_bodies[(self._bar_count - 1 - i) % 5] for i in range(3)) / 3
        else:
            momentum_3 = change
            delta_sum_3 = bar.delta
            bullish_3 = 0.5

        # 직전 5바 모멘텀
        if n >= 5:
            closes_5 = [self._recent_closes[(self._bar_count - 1 - i) % 5] for i in range(5)]
            momentum_5 = closes_5[0] - closes_5[4]
            delta_sum_5 = sum(self._recent_deltas[(self._bar_count - 1 - i) % 5] for i in range(5))
            avg_range_5 = np.mean([self._recent_ranges[(self._bar_count - 1 - i) % 5] for i in range(5)])
        else:
            momentum_5 = momentum_3
            delta_sum_5 = delta_sum_3
            avg_range_5 = bar.range

        range_ratio = bar.range / avg_range_5 if avg_range_5 > 0 else 1.0

        # ── 개장 직후 특화 피처 ──
        # 갭
        gap = 0.0
        if self._prev_day_close > 0:
            if self._session_bar_count == 1:
                self._session_first_close = bar.open
            gap = self._session_first_close - self._prev_day_close

        # 갭 채움 비율
        gap_fill = 0.0
        if abs(gap) > 0.25:
            if gap > 0:  # 갭 업
                filled = self._session_first_close - self._day_low
                gap_fill = min(filled / gap, 1.0)
            else:  # 갭 다운
                filled = self._day_high - self._session_first_close
                gap_fill = min(filled / abs(gap), 1.0)

        # 경과 시간 (분)
        minutes_since_open = 0.0
        if self._session_start_ns > 0 and bar.timestamp_ns > 0:
            minutes_since_open = (bar.timestamp_ns - self._session_start_ns) / 60e9

        # 오프닝 레인지 (첫 ~20바 = 약 5분)
        if self._session_bar_count <= 20:
            self._opening_range_high = max(self._opening_range_high, bar.high)
            self._opening_range_low = min(self._opening_range_low, bar.low)
        elif not self._opening_range_set:
            self._opening_range_set = True

        or_range = self._opening_range_high - self._opening_range_low
        or_position = 0.5
        if or_range > 0 and self._opening_range_set:
            or_position = (close - self._opening_range_low) / or_range

        # 세션 방향성 (효율 비율)
        session_net_move = close - self._session_first_close if self._session_first_close > 0 else 0.0
        session_total_move = sum(
            abs(self._recent_closes[(self._bar_count - 1 - i) % 5]
                - self._recent_closes[(self._bar_count - 2 - i) % 5])
            for i in range(min(n - 1, 4))
        ) if n >= 2 else 0.001
        session_direction = session_net_move / session_total_move if session_total_move > 0 else 0.0

        # ── 피처 배열 기록 ──
        f = self.features
        # 가격 구조
        f[0] = close
        f[1] = change
        f[2] = self._ema_fast
        f[3] = self._ema_slow
        f[4] = self._ema_fast - self._ema_slow
        f[5] = rsi
        f[6] = bar.range
        f[7] = bar.body
        f[8] = bar.close_position
        # 주문 흐름
        f[9] = bar.delta
        f[10] = bar.buy_ticks / bar.tick_count if bar.tick_count > 0 else 0.5
        f[11] = self._cum_delta
        f[12] = bar.duration_ms
        # 일중 컨텍스트
        f[13] = day_position
        f[14] = self._day_high
        f[15] = self._day_low
        f[16] = close - self._ema_fast
        f[17] = close - self._ema_slow
        # 시퀀스 대체
        f[18] = momentum_3
        f[19] = momentum_5
        f[20] = delta_sum_3
        f[21] = delta_sum_5
        f[22] = bullish_3
        f[23] = range_ratio
        # 개장 직후 특화
        f[24] = gap
        f[25] = gap_fill
        f[26] = minutes_since_open
        f[27] = or_position
        f[28] = session_direction
        f[29] = session_net_move

        self._prev_close = close

    def reset_daily(self) -> None:
        self._day_high = -np.inf
        self._day_low = np.inf
        self._cum_delta = 0
        self._opening_range_high = 0.0
        self._opening_range_low = np.inf
        self._opening_range_set = False
        self._session_first_close = 0.0
        self._session_bar_count = 0

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "close", "change", "ema_fast", "ema_slow", "macd", "rsi",
            "bar_range", "bar_body", "close_position",
            "delta", "buy_ratio", "cum_delta", "duration_ms",
            "day_position", "day_high", "day_low",
            "dist_ema_fast", "dist_ema_slow",
            "momentum_3bar", "momentum_5bar",
            "delta_sum_3bar", "delta_sum_5bar",
            "bullish_ratio_3bar", "range_ratio",
            "gap", "gap_fill_ratio", "minutes_since_open",
            "opening_range_pos", "session_direction", "session_momentum",
        ]


class AIStrategy(BaseStrategy):
    """
    LightGBM 기반 진입 전략.

    사전 준비 (Pre-inference) 흐름:
      1. 매 틱 → on_tick_update() 호출
      2. 바 채움 비율이 threshold(96%) 이상이면
         → 현재 상태로 예비 추론 실행, 결과 캐싱
      3. 바 완성 → on_bar() 호출
         → 최종 틱 반영 후 캐시된 결과와 비교
         → 변화 미미하면 캐시 결과 즉시 사용 (추론 스킵)
         → 변화 크면 최종 추론 1회 실행
    """

    name = "ai"

    def __init__(self, config: dict):
        self.min_confidence = config.get("min_confidence", 0.65)
        self.tp_points = config.get("take_profit_points", 15)
        self.sl_points = config.get("stop_loss_points", 10)
        self.pre_inference = config.get("ai", {}).get("pre_inference", True)
        self.pre_warm_threshold = config.get("pre_warm_threshold", 0.96)

        self._features = IncrementalFeatures()
        self._model = None
        self._pre_cache: _PreCache | None = None

    def load_model(self, model_path: str) -> None:
        """모델을 메모리에 로드하고 워밍업."""
        try:
            import lightgbm as lgb
            self._model = lgb.Booster(model_file=model_path)
            dummy = np.zeros((1, IncrementalFeatures.FEATURE_COUNT))
            for _ in range(100):
                self._model.predict(dummy)
            log.info("AI model loaded and warmed up: %s", model_path)
        except FileNotFoundError:
            log.warning("Model file not found: %s — running without AI", model_path)
            self._model = None

    def warmup(self, bars: list[Bar]) -> None:
        self._features.warmup(bars)
        log.info("AI strategy warmed up with %d bars", len(bars))

    def on_tick_update(self, bars: list[Bar], buffer_ratio: float) -> None:
        if not self.pre_inference or self._model is None:
            return
        if buffer_ratio < self.pre_warm_threshold:
            return

        current_bar = bars[-1] if bars else None
        if current_bar is None:
            return

        preview_features = self._features.preview(current_bar)
        prob = self._predict(preview_features)

        self._pre_cache = _PreCache(
            probability=prob,
            features_snapshot=preview_features.copy(),
            timestamp_ns=time.time_ns(),
        )

    def on_bar(self, bars: list[Bar]) -> Signal | None:
        if self._model is None:
            return None

        new_bar = bars[-1]
        features = self._features.update(new_bar)

        if self._pre_cache is not None:
            drift = np.max(np.abs(features - self._pre_cache.features_snapshot))
            if drift < 0.5:
                prob = self._pre_cache.probability
            else:
                prob = self._predict(features)
            self._pre_cache = None
        else:
            prob = self._predict(features)

        if prob > (0.5 + self.min_confidence / 2):
            direction = Direction.LONG
            confidence = (prob - 0.5) * 2
        elif prob < (0.5 - self.min_confidence / 2):
            direction = Direction.SHORT
            confidence = (0.5 - prob) * 2
        else:
            return None

        return Signal(
            direction=direction,
            confidence=confidence,
            take_profit=self.tp_points,
            stop_loss=self.sl_points,
            strategy_name=self.name,
            metadata={"probability": prob, "used_cache": self._pre_cache is None},
        )

    def _predict(self, features: np.ndarray) -> float:
        return self._model.predict(
            features.reshape(1, -1),
            num_threads=1,
        )[0]

    def reset(self) -> None:
        self._features.reset_daily()
        self._pre_cache = None


class _PreCache:
    __slots__ = ("probability", "features_snapshot", "timestamp_ns")

    def __init__(self, probability: float, features_snapshot: np.ndarray, timestamp_ns: int):
        self.probability = probability
        self.features_snapshot = features_snapshot
        self.timestamp_ns = timestamp_ns
