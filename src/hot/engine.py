"""
메인 트레이딩 엔진.

세션 흐름:
  09:30 ET  시스템 시작 → 브로커 연결, 틱 수집 시작
  09:30~35  WARMUP — 틱 데이터 수집 + 전략 웜업 (거래 안 함)
  09:35     ACTIVE — 거래 시작
  ~10:00    (지표 있으면) BLACKOUT — 포지션 청산, 거래 중지
  ~10:10    (지표 끝나면) ACTIVE 복귀
  10:29:30  포지션 전체 청산
  10:30     DONE — 종료

핫패스 이벤트 루프:
  틱 수신 → 세션 체크 → 차트 업데이트 → (예비 추론) → 전략 평가 → 리스크 체크 → 주문
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Protocol

from src.cold.tick_recorder import TickRecorder
from src.hot.executor import ExitManager, OrderExecutor
from src.hot.risk_gate import RiskGate
from src.hot.session import SessionManager, SessionPhase
from src.hot.tick_chart import Tick, TickChart
from src.strategy.base import Bar
from src.strategy.manager import StrategyManager

log = logging.getLogger(__name__)


class BrokerAdapter(Protocol):
    async def connect(self) -> None: ...
    async def subscribe_ticks(self, symbol: str) -> None: ...
    async def next_tick(self) -> Tick: ...
    async def place_order(self, direction: int, quantity: int, price: float) -> float: ...
    async def close_position(self, direction: int, price: float) -> None: ...
    async def disconnect(self) -> None: ...


class EventBuffer:
    def __init__(self, maxsize: int = 10000):
        self._buffer: list[dict[str, Any]] = [{}] * maxsize
        self._size = maxsize
        self._write_idx = 0

    def push(self, event: dict[str, Any]) -> None:
        self._buffer[self._write_idx % self._size] = event
        self._write_idx += 1


class TradingEngine:

    def __init__(self, config: dict, broker: BrokerAdapter):
        self.config = config
        self.broker = broker

        # 세션 관리자
        self.session = SessionManager(config)

        # 핫패스 컴포넌트
        chart_cfg = config.get("tick_chart", {})
        self.tick_chart = TickChart(
            tick_count=chart_cfg.get("size", 500),
            adaptive=chart_cfg.get("adaptive", False),
            adaptive_config=chart_cfg.get("adaptive_config"),
        )
        self.strategy_manager = StrategyManager(config.get("strategy", {}))
        self.risk_gate = RiskGate(config)
        self.exit_manager = ExitManager(config)
        self.executor = OrderExecutor(broker, self.exit_manager)

        self.events = EventBuffer()

        # 틱 자동 저장 (콜드패스, 별도 스레드)
        self.tick_recorder = TickRecorder(
            output_dir=config.get("data", {}).get("tick_dir", "data/raw"),
        )

        self._running = False
        self._last_price = 0.0
        self._heartbeat_ns = 0
        self._warmup_done = False

    async def start(self) -> None:
        log.info("Engine starting...")
        await self.broker.connect()
        await self.broker.subscribe_ticks(
            self.config.get("broker", {}).get("contract", {}).get("symbol", "NQ")
        )
        self._running = True
        self.tick_recorder.start()
        log.info("Engine started — waiting for session open")
        await self._main_loop()

    async def stop(self) -> None:
        self._running = False
        self.tick_recorder.stop()
        if self.executor.positions:
            log.warning("Flattening all positions on shutdown")
            await self.executor.flatten_all(self._last_price)
        await self.broker.disconnect()

        # 일일 리포트
        stats = self.risk_gate.stats
        log.info(
            "SESSION CLOSED | PnL=$%.2f | Trades=%d",
            stats["daily_pnl"], stats["trade_count"],
        )

    async def _main_loop(self) -> None:
        while self._running:
            try:
                tick = await self.broker.next_tick()
                self._last_price = tick.price
                self._heartbeat_ns = time.time_ns()

                # 틱 자동 저장 (deque.append — 논블로킹)
                self.tick_recorder.on_tick(
                    tick.timestamp_ns, tick.price, tick.size, tick.side,
                )

                # ── 1. 세션 상태 체크 ──
                phase = self.session.update()

                if phase == SessionPhase.PRE_MARKET:
                    continue

                if phase == SessionPhase.DONE:
                    if self.executor.positions:
                        await self.executor.flatten_all(tick.price)
                        self._record_flatten_all()
                    await self.stop()
                    return

                if phase == SessionPhase.BLACKOUT:
                    # 블랙아웃: 신규 진입 차단, 기존 포지션은 조건부 청산
                    # - 수익 중 → TP 도달 시 청산, 진입가 되돌림 시 본전 탈출
                    # - 손실 중 → 최초 SL 도달 시 시장가 탈출
                    if self.executor.positions:
                        closed = await self.executor.check_exits_blackout(tick.price)
                        for pos, action, pnl in closed:
                            pnl_usd = pnl * 5.0
                            self.risk_gate.on_close(pnl_usd)
                            self.events.push({
                                "type": "exit",
                                "action": action.name,
                                "pnl_points": pnl,
                                "pnl_usd": pnl_usd,
                                "reason": "blackout",
                                "timestamp_ns": time.time_ns(),
                            })
                    # 틱 차트는 계속 업데이트 (블랙아웃 후 ACTIVE 복귀 대비)
                    self.tick_chart.on_tick(tick)
                    continue

                # ── 2. 틱 차트 업데이트 (WARMUP + ACTIVE 공통) ──
                bar = self.tick_chart.on_tick(tick)

                # WARMUP: 데이터만 수집
                if phase == SessionPhase.WARMUP:
                    # 바가 충분히 쌓이면 전략 웜업
                    if bar is not None and not self._warmup_done:
                        bars_list = self.tick_chart.bars_as_list()
                        if len(bars_list) >= 5:
                            self.strategy_manager.warmup(bars_list)
                            self._warmup_done = True
                            log.info("Strategy warmup complete (%d bars)", len(bars_list))
                    continue

                # ── 3. ACTIVE: 거래 로직 ──

                # 포지션 청산 체크 (매 틱)
                if self.executor.positions:
                    closed = await self.executor.check_exits(tick.price)
                    for pos, action, pnl in closed:
                        pnl_usd = pnl * 5.0
                        self.risk_gate.on_close(pnl_usd)
                        self.events.push({
                            "type": "exit",
                            "action": action.name,
                            "pnl_points": pnl,
                            "pnl_usd": pnl_usd,
                            "timestamp_ns": time.time_ns(),
                        })

                # 예비 추론 (바 완성 전)
                bars_list = self.tick_chart.bars_as_list()
                ratio = self.tick_chart.buffer_ratio

                if ratio > 0 and bar is None:
                    partial = self.tick_chart.current_partial_bar
                    if partial is not None:
                        preview_bars = bars_list + [partial]
                        self.strategy_manager.on_tick_update(preview_bars, ratio)

                # 바 완성 → 전략 평가
                if bar is not None:
                    signal = self.strategy_manager.evaluate(bars_list)

                    if signal is not None and self.risk_gate.allow(signal):
                        # 거래 종료 임박 시 신규 진입 차단 (마지막 3분)
                        if self.session.time_remaining_sec() < 180:
                            log.info("Skip entry — session ending in < 3 min")
                            continue

                        position = await self.executor.enter(signal, tick.price)
                        if position is not None:
                            self.risk_gate.on_fill(1)
                            self.events.push({
                                "type": "entry",
                                "direction": signal.direction.name,
                                "price": tick.price,
                                "confidence": signal.confidence,
                                "strategy": signal.strategy_name,
                                "regime": signal.metadata.get("regime", ""),
                                "timestamp_ns": time.time_ns(),
                            })

            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Error in main loop")

    def _record_flatten_all(self) -> None:
        self.events.push({
            "type": "flatten_all",
            "reason": self.session.phase.name,
            "timestamp_ns": time.time_ns(),
        })
