"""
Interactive Brokers 어댑터.
ib_insync를 사용하여 BrokerAdapter 인터페이스를 구현.
"""

from __future__ import annotations

import asyncio
import logging
import socket

from src.hot.tick_chart import Tick

log = logging.getLogger(__name__)


class IBKRAdapter:
    """
    IBKR TWS/Gateway 연결 어댑터.

    사용법:
        adapter = IBKRAdapter(config["broker"])
        await adapter.connect()
        await adapter.subscribe_ticks("NQ")
        tick = await adapter.next_tick()
    """

    def __init__(self, config: dict):
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 7497)
        self.client_id = config.get("client_id", 1)
        self.contract_cfg = config.get("contract", {})

        self._ib = None       # ib_insync.IB 인스턴스
        self._contract = None
        self._tick_queue: asyncio.Queue[Tick] = asyncio.Queue(maxsize=50000)

    async def connect(self) -> None:
        from ib_insync import IB, Future

        self._ib = IB()
        await self._ib.connectAsync(
            host=self.host,
            port=self.port,
            clientId=self.client_id,
        )
        # TCP_NODELAY 설정
        sock = self._ib.client._socket
        if sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # 계약 설정
        self._contract = Future(
            symbol=self.contract_cfg.get("symbol", "NQ"),
            exchange=self.contract_cfg.get("exchange", "CME"),
            lastTradeDateOrContractMonth=self.contract_cfg.get("expiry", ""),
        )
        qualified = await self._ib.qualifyContractsAsync(self._contract)
        if qualified:
            self._contract = qualified[0]
        log.info("Connected to IBKR — contract: %s", self._contract)

    async def subscribe_ticks(self, symbol: str) -> None:
        import time

        self._ib.reqTickByTickData(
            self._contract, tickType="AllLast", numberOfTicks=0, ignoreSize=True
        )

        def on_tick_by_tick(ticker, tick_data):
            try:
                t = Tick(
                    price=tick_data.price,
                    size=tick_data.size,
                    side=1 if tick_data.price >= ticker.marketPrice() else -1,
                    timestamp_ns=int(tick_data.time.timestamp() * 1e9)
                    if tick_data.time
                    else int(time.time() * 1e9),
                )
                self._tick_queue.put_nowait(t)
            except asyncio.QueueFull:
                pass  # 큐 가득 차면 오래된 틱은 드롭 (최신 유지)

        self._ib.pendingTickersEvent += on_tick_by_tick
        log.info("Subscribed to tick data: %s", symbol)

    async def next_tick(self) -> Tick:
        return await self._tick_queue.get()

    async def place_order(self, direction: int, quantity: int, price: float) -> float:
        from ib_insync import MarketOrder

        action = "BUY" if direction == 1 else "SELL"
        order = MarketOrder(action, quantity)
        order.tif = "IOC"

        trade = self._ib.placeOrder(self._contract, order)

        # 체결 대기 (최대 5초)
        for _ in range(50):
            if trade.orderStatus.status == "Filled":
                fill_price = trade.orderStatus.avgFillPrice
                log.info("Order filled: %s %d @ %.2f", action, quantity, fill_price)
                return fill_price
            await asyncio.sleep(0.1)

        # 타임아웃 시 시장가로 가정
        log.warning("Order fill timeout — using submitted price %.2f", price)
        return price

    async def close_position(self, direction: int, price: float) -> None:
        # 반대 방향으로 시장가 주문
        close_direction = -direction
        await self.place_order(close_direction, 1, price)

    async def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            log.info("Disconnected from IBKR")
