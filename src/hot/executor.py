"""
주문 실행기.
주문 객체를 사전 생성(풀링)하여 진입 시 지연 최소화.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto

from src.strategy.base import Direction, Signal

log = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = auto()
    SUBMITTED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()


class ExitAction(Enum):
    TAKE_PROFIT = auto()
    STOP_LOSS = auto()
    TRAILING_STOP = auto()
    TIME_EXIT = auto()
    BLACKOUT_BREAKEVEN = auto()    # 블랙아웃 중 진입가 도달 → 본전 탈출
    BLACKOUT_TAKE_PROFIT = auto()  # 블랙아웃 중 수익실현
    BLACKOUT_STOP_LOSS = auto()    # 블랙아웃 중 손절


@dataclass
class Position:
    direction: Direction
    entry_price: float
    take_profit_price: float       # 최초 진입 시 설정된 수익실현 가격
    stop_loss_price: float         # 최초 진입 시 설정된 손절 가격
    quantity: int = 1
    highest_pnl: float = 0.0

    def unrealized_pnl_points(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.direction

    def update_highest(self, current_price: float) -> None:
        pnl = self.unrealized_pnl_points(current_price)
        if pnl > self.highest_pnl:
            self.highest_pnl = pnl


class ExitManager:
    """수익실현 / 손절 / 트레일링 스탑 관리."""

    def __init__(self, config: dict):
        exit_cfg = config.get("exit", config)
        self.tp_points = exit_cfg.get("take_profit_points", 15)
        self.sl_points = exit_cfg.get("stop_loss_points", 10)

        trailing = exit_cfg.get("trailing", {})
        self.trailing_enabled = trailing.get("enabled", True)
        self.trailing_trigger = trailing.get("trigger_points", 8)
        self.trailing_offset = trailing.get("offset_points", 4)

    def check(self, position: Position, current_price: float) -> ExitAction | None:
        """통상 거래 시간(ACTIVE)의 청산 판단."""
        pnl = position.unrealized_pnl_points(current_price)
        position.update_highest(current_price)

        # 손절
        if pnl <= -self.sl_points:
            return ExitAction.STOP_LOSS

        # 수익실현
        if pnl >= self.tp_points:
            return ExitAction.TAKE_PROFIT

        # 트레일링 스탑
        if self.trailing_enabled and pnl >= self.trailing_trigger:
            trailing_level = position.highest_pnl - self.trailing_offset
            if pnl <= trailing_level:
                return ExitAction.TRAILING_STOP

        return None

    def check_blackout(self, position: Position, current_price: float) -> ExitAction | None:
        """
        블랙아웃 구간의 청산 판단.

        수익 중:
          - 수익실현가(TP) 도달 → 청산
          - 진입가까지 되돌림   → 시장가 탈출 (본전 방어)

        손실 중:
          - 최초 손절가(SL) 도달 → 시장가 탈출
        """
        pnl = position.unrealized_pnl_points(current_price)

        if pnl > 0:
            # 수익 중 — TP 도달 체크
            if position.direction == Direction.LONG:
                if current_price >= position.take_profit_price:
                    return ExitAction.BLACKOUT_TAKE_PROFIT
                # 진입가까지 하락 → 본전 탈출
                if current_price <= position.entry_price:
                    return ExitAction.BLACKOUT_BREAKEVEN
            else:  # SHORT
                if current_price <= position.take_profit_price:
                    return ExitAction.BLACKOUT_TAKE_PROFIT
                # 진입가까지 상승 → 본전 탈출
                if current_price >= position.entry_price:
                    return ExitAction.BLACKOUT_BREAKEVEN

        else:
            # 손실 중 — 최초 SL 도달 체크
            if position.direction == Direction.LONG:
                if current_price <= position.stop_loss_price:
                    return ExitAction.BLACKOUT_STOP_LOSS
            else:  # SHORT
                if current_price >= position.stop_loss_price:
                    return ExitAction.BLACKOUT_STOP_LOSS

        return None


class OrderExecutor:
    """
    브로커 API와의 주문 인터페이스.
    실제 브로커 연결은 broker 어댑터를 주입받아 사용.
    """

    def __init__(self, broker_adapter, exit_manager: ExitManager):
        self.broker = broker_adapter
        self.exit_manager = exit_manager
        self.positions: list[Position] = []

    async def enter(self, signal: Signal, current_price: float) -> Position | None:
        """신호에 따라 주문 실행. 진입 시 TP/SL 가격을 계산하여 Position에 저장."""
        try:
            fill_price = await self.broker.place_order(
                direction=signal.direction,
                quantity=1,
                price=current_price,
            )

            # 최초 진입 시 TP/SL 절대 가격 계산
            if signal.direction == Direction.LONG:
                tp_price = fill_price + signal.take_profit
                sl_price = fill_price - signal.stop_loss
            else:
                tp_price = fill_price - signal.take_profit
                sl_price = fill_price + signal.stop_loss

            position = Position(
                direction=signal.direction,
                entry_price=fill_price,
                take_profit_price=tp_price,
                stop_loss_price=sl_price,
            )
            self.positions.append(position)
            log.info(
                "ENTRY %s @ %.2f | TP=%.2f SL=%.2f (conf=%.2f, strategy=%s)",
                "LONG" if signal.direction == Direction.LONG else "SHORT",
                fill_price, tp_price, sl_price,
                signal.confidence, signal.strategy_name,
            )
            return position
        except Exception:
            log.exception("Order execution failed")
            return None

    async def check_exits(self, current_price: float) -> list[tuple[Position, ExitAction, float]]:
        """통상 거래 시간의 청산 체크."""
        return await self._check_with(
            current_price,
            self.exit_manager.check,
        )

    async def check_exits_blackout(self, current_price: float) -> list[tuple[Position, ExitAction, float]]:
        """블랙아웃 구간의 청산 체크."""
        return await self._check_with(
            current_price,
            self.exit_manager.check_blackout,
        )

    async def _check_with(self, current_price, check_fn) -> list[tuple[Position, ExitAction, float]]:
        closed = []
        remaining = []

        for pos in self.positions:
            action = check_fn(pos, current_price)
            if action is not None:
                pnl = pos.unrealized_pnl_points(current_price)
                await self.broker.close_position(pos.direction, current_price)
                log.info(
                    "EXIT %s @ %.2f | %s | PnL=%.1f pts (entry=%.2f)",
                    "LONG" if pos.direction == Direction.LONG else "SHORT",
                    current_price, action.name, pnl, pos.entry_price,
                )
                closed.append((pos, action, pnl))
            else:
                remaining.append(pos)

        self.positions = remaining
        return closed

    async def flatten_all(self, current_price: float) -> None:
        """전체 포지션 즉시 청산 (세션 종료/비상용)."""
        for pos in self.positions:
            pnl = pos.unrealized_pnl_points(current_price)
            await self.broker.close_position(pos.direction, current_price)
            log.warning(
                "FLATTEN %s @ %.2f | PnL=%.1f pts (entry=%.2f)",
                pos.direction.name, current_price, pnl, pos.entry_price,
            )
        self.positions.clear()
