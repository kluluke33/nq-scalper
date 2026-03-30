"""
커스텀 전략 템플릿.

새 전략을 추가하려면:
  1. 이 파일을 복사하여 새 파일 생성 (예: mean_reversion_strategy.py)
  2. 클래스명과 name 변경
  3. on_bar() 구현
  4. @register_strategy 데코레이터 추가
  5. config/settings.yaml에서 active 전략으로 설정하거나 composite에 추가
"""

from __future__ import annotations

from src.strategy.base import Bar, BaseStrategy, Direction, Signal
from src.strategy.manager import register_strategy


@register_strategy
class CustomTemplate(BaseStrategy):

    name = "custom_template"

    def __init__(self, config: dict):
        self.tp_points = config.get("take_profit_points", 15)
        self.sl_points = config.get("stop_loss_points", 10)

    def warmup(self, bars: list[Bar]) -> None:
        """과거 데이터로 초기화. 필요 없으면 pass."""
        pass

    def on_bar(self, bars: list[Bar]) -> Signal | None:
        """
        새 바 완성 시 호출.
        진입 신호가 있으면 Signal 반환, 없으면 None.
        """
        # 여기에 전략 로직 구현
        # 예시:
        #   if some_condition(bars):
        #       return Signal(
        #           direction=Direction.LONG,
        #           confidence=0.8,
        #           take_profit=self.tp_points,
        #           stop_loss=self.sl_points,
        #           strategy_name=self.name,
        #       )
        return None

    def on_tick_update(self, bars: list[Bar], buffer_ratio: float) -> None:
        """
        (선택) 매 틱마다 호출.
        사전 준비가 필요한 전략만 구현.
        buffer_ratio: 바 채움 비율 (0.0~1.0), 1.0 근접 시 바 완성 임박.
        """
        pass

    def reset(self) -> None:
        """일일 리셋. 필요한 상태 초기화."""
        pass
