"""
실시간 틱 자동 저장.

매일 거래 시간(09:30~10:30 ET) 동안 수신되는 틱을 CSV로 저장.
엔진의 콜드패스에서 실행 — 핫패스 성능에 영향 없음.

저장 형식: data/raw/ticks_YYYYMMDD.csv
  → 기존 Databento 데이터와 동일한 형식
  → 모델 재학습 시 Databento + IBKR 데이터를 합쳐서 사용

사용 방식:
  TradingEngine 시작 시 TickRecorder를 등록하면
  매 틱마다 on_tick()이 호출되어 버퍼에 쌓이고,
  주기적으로 파일에 flush.
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

ET = timezone(timedelta(hours=-4))


class TickRecorder:
    """
    실시간 틱 → CSV 저장.
    쓰기는 별도 스레드에서 수행하여 핫패스 차단 없음.
    """

    def __init__(self, output_dir: str = "data/raw", flush_interval_sec: float = 5.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flush_interval = flush_interval_sec

        self._buffer: deque = deque()
        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self._current_date = ""
        self._tick_count = 0
        self._running = False
        self._flush_thread: threading.Thread | None = None

    def start(self) -> None:
        """저장 시작. flush 스레드 가동."""
        self._running = True
        self._open_file()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        log.info("TickRecorder started — saving to %s", self.output_dir)

    def stop(self) -> None:
        """저장 중지. 버퍼 flush 후 파일 닫기."""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=3)
        self._flush_buffer()
        self._close_file()
        log.info("TickRecorder stopped — %d ticks saved today", self._tick_count)

    def on_tick(self, timestamp_ns: int, price: float, size: int, side: int) -> None:
        """
        매 틱마다 호출 (핫패스에서).
        deque.append는 thread-safe이므로 lock 불필요.
        """
        self._buffer.append((timestamp_ns, price, size, side))

    def _flush_loop(self) -> None:
        """별도 스레드에서 주기적으로 버퍼를 파일에 기록."""
        while self._running:
            time.sleep(self.flush_interval)
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """버퍼의 모든 틱을 파일에 기록."""
        if not self._buffer:
            return

        # 날짜 변경 체크
        today = datetime.now(ET).strftime("%Y%m%d")
        if today != self._current_date:
            self._close_file()
            self._current_date = today
            self._tick_count = 0
            self._open_file()

        # 버퍼에서 꺼내서 기록
        ticks_to_write = []
        while self._buffer:
            try:
                ticks_to_write.append(self._buffer.popleft())
            except IndexError:
                break

        if ticks_to_write and self._writer:
            for ts_ns, price, size, side in ticks_to_write:
                ts = datetime.fromtimestamp(ts_ns / 1e9, tz=ET)
                self._writer.writerow([
                    ts.isoformat(),
                    price,
                    size,
                    side,
                ])
            self._file.flush()
            self._tick_count += len(ticks_to_write)

    def _open_file(self) -> None:
        if not self._current_date:
            self._current_date = datetime.now(ET).strftime("%Y%m%d")

        file_path = self.output_dir / f"ticks_{self._current_date}.csv"
        file_exists = file_path.exists()

        self._file = open(file_path, "a", newline="")
        self._writer = csv.writer(self._file)

        if not file_exists:
            self._writer.writerow(["timestamp", "price", "size", "side"])
            self._file.flush()

    def _close_file(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
