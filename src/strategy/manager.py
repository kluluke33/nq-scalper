"""
전략 매니저.

역할:
  - 전략 등록/제거 (플러그인 방식)
  - 단일 전략, 복수 전략 합성, 시장 상태 기반 자동 전환 지원
  - 엔진에 단일 인터페이스 제공: evaluate()

전략 결정 방식 (active 설정):
  "ai"         → AI 전략만 사용 (고정)
  "breakout"   → 브레이크아웃 전략만 사용 (고정)
  "composite"  → 복수 전략의 신호를 합성
  "auto"       → 시장 상태(regime)에 따라 자동 전환
"""

from __future__ import annotations

import logging

from src.strategy.base import Bar, BaseStrategy, Direction, Signal
from src.strategy.market_regime import MarketRegime, RegimeDetector

log = logging.getLogger(__name__)

# 전략 클래스 레지스트리 — @register_strategy 데코레이터로 자동 등록
_STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register_strategy(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    """
    전략 클래스를 레지스트리에 등록하는 데코레이터.

    사용법:
        @register_strategy
        class MyStrategy(BaseStrategy):
            name = "my_strategy"
            ...
    """
    _STRATEGY_REGISTRY[cls.name] = cls
    log.info("Strategy registered: %s", cls.name)
    return cls


def get_registered_strategies() -> dict[str, type[BaseStrategy]]:
    return dict(_STRATEGY_REGISTRY)


class CombineMode:
    UNANIMOUS = "unanimous"    # 모든 전략이 같은 방향일 때만
    ANY = "any"                # 하나라도 신호 발생 시
    WEIGHTED = "weighted"      # 가중 평균 확신도 기반


class StrategyManager:

    def __init__(self, config: dict):
        self._strategies: dict[str, BaseStrategy] = {}
        self._active_name = config.get("active", "ai")
        self._combine_mode = CombineMode.UNANIMOUS
        self._weights: dict[str, float] = {}
        self._min_combined_score = 0.5

        # 복합 전략 설정
        composite_cfg = config.get("composite", {})
        if self._active_name == "composite":
            self._combine_mode = composite_cfg.get("mode", CombineMode.UNANIMOUS)
            self._weights = composite_cfg.get("weights", {})
            self._min_combined_score = composite_cfg.get("min_combined_score", 0.6)

        # 자동 전환 설정
        self._regime_detector = RegimeDetector()
        self._regime_strategy_map: dict[MarketRegime, str] = {}
        if self._active_name == "auto":
            auto_cfg = config.get("auto", {})
            self._regime_strategy_map = {
                MarketRegime.TRENDING:  auto_cfg.get("trending", "ai"),
                MarketRegime.RANGING:   auto_cfg.get("ranging", "breakout"),
                MarketRegime.VOLATILE:  auto_cfg.get("volatile", "ai"),
                MarketRegime.QUIET:     auto_cfg.get("quiet", "_skip"),
            }
        self._current_regime = MarketRegime.RANGING

    def add(self, strategy: BaseStrategy) -> None:
        self._strategies[strategy.name] = strategy
        log.info("Strategy added: %s", strategy.name)

    def remove(self, name: str) -> None:
        self._strategies.pop(name, None)
        log.info("Strategy removed: %s", name)

    def warmup(self, bars: list[Bar]) -> None:
        for s in self._strategies.values():
            s.warmup(bars)

    def on_tick_update(self, bars: list[Bar], buffer_ratio: float) -> None:
        """모든 등록된 전략에 틱 업데이트 전파."""
        for s in self._strategies.values():
            s.on_tick_update(bars, buffer_ratio)

    def evaluate(self, bars: list[Bar]) -> Signal | None:
        """
        설정된 모드에 따라 신호를 생성.
        - 고정 단일 전략: 해당 전략의 on_bar() 결과 반환
        - composite: 복수 전략 합성
        - auto: 시장 상태 판별 후 적합한 전략 자동 선택
        """
        if self._active_name == "auto":
            return self._evaluate_auto(bars)
        elif self._active_name == "composite":
            return self._evaluate_composite(bars)
        else:
            return self._evaluate_single(self._active_name, bars)

    def _evaluate_single(self, name: str, bars: list[Bar]) -> Signal | None:
        strategy = self._strategies.get(name)
        if strategy is None:
            log.warning("Strategy '%s' not found", name)
            return None
        return strategy.on_bar(bars)

    def _evaluate_auto(self, bars: list[Bar]) -> Signal | None:
        """시장 상태를 판별하고 해당하는 전략을 실행."""
        self._current_regime = self._regime_detector.update(bars)

        if self._regime_detector.changed:
            log.info("Market regime changed → %s", self._current_regime.name)

        strategy_name = self._regime_strategy_map.get(self._current_regime, "_skip")

        # _skip = 진입하지 않음 (QUIET 등)
        if strategy_name == "_skip":
            return None

        signal = self._evaluate_single(strategy_name, bars)

        # 시장 상태에 따른 확신도 보정
        if signal is not None:
            signal = self._adjust_for_regime(signal)
            signal.metadata["regime"] = self._current_regime.name

        return signal

    def _adjust_for_regime(self, signal: Signal) -> Signal:
        """시장 상태에 따라 확신도 임계값과 TP/SL을 보정."""
        if self._current_regime == MarketRegime.VOLATILE:
            # 변동성 높을 때: 확신도 기준 강화, 손절 확대
            signal.confidence *= 0.8
            signal.stop_loss *= 1.5
        elif self._current_regime == MarketRegime.TRENDING:
            # 추세장: 수익 목표 확대
            signal.take_profit *= 1.3
        return signal

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    def _evaluate_composite(self, bars: list[Bar]) -> Signal | None:
        """복수 전략의 신호를 합성."""
        composite_cfg_strategies = list(self._weights.keys()) or list(self._strategies.keys())
        signals: list[Signal] = []

        for name in composite_cfg_strategies:
            strategy = self._strategies.get(name)
            if strategy is None:
                continue
            sig = strategy.on_bar(bars)
            if sig is not None:
                signals.append(sig)

        if not signals:
            return None

        if self._combine_mode == CombineMode.UNANIMOUS:
            return self._combine_unanimous(signals)
        elif self._combine_mode == CombineMode.ANY:
            return self._combine_any(signals)
        elif self._combine_mode == CombineMode.WEIGHTED:
            return self._combine_weighted(signals)
        return None

    def _combine_unanimous(self, signals: list[Signal]) -> Signal | None:
        if len(signals) < 2:
            return None
        direction = signals[0].direction
        if not all(s.direction == direction for s in signals):
            return None
        avg_conf = sum(s.confidence for s in signals) / len(signals)
        return Signal(
            direction=direction,
            confidence=avg_conf,
            take_profit=min(s.take_profit for s in signals),
            stop_loss=max(s.stop_loss for s in signals),
            strategy_name="composite_unanimous",
            metadata={"sources": [s.strategy_name for s in signals]},
        )

    def _combine_any(self, signals: list[Signal]) -> Signal | None:
        return max(signals, key=lambda s: s.confidence)

    def _combine_weighted(self, signals: list[Signal]) -> Signal | None:
        total_weight = 0.0
        weighted_score = 0.0

        for sig in signals:
            w = self._weights.get(sig.strategy_name, 1.0)
            weighted_score += sig.direction * sig.confidence * w
            total_weight += w

        if total_weight == 0:
            return None

        normalized = weighted_score / total_weight

        if abs(normalized) < self._min_combined_score:
            return None

        direction = Direction.LONG if normalized > 0 else Direction.SHORT

        return Signal(
            direction=direction,
            confidence=abs(normalized),
            take_profit=signals[0].take_profit,
            stop_loss=signals[0].stop_loss,
            strategy_name="composite_weighted",
            metadata={"weighted_score": normalized, "sources": [s.strategy_name for s in signals]},
        )

    def reset(self) -> None:
        for s in self._strategies.values():
            s.reset()
