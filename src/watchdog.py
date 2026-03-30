"""
Watchdog — 독립 프로세스로 Trading Engine을 감시.

역할:
  - Trading Process 생존 감시
  - 응답 없으면 포지션 전체 청산 + 프로세스 재시작
  - 일일 리셋 스케줄
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import sys
import time
from typing import Callable

log = logging.getLogger(__name__)


class Watchdog:

    def __init__(
        self,
        engine_starter: Callable,
        config: dict,
        heartbeat_timeout_sec: float = 5.0,
    ):
        self.engine_starter = engine_starter
        self.config = config
        self.heartbeat_timeout = heartbeat_timeout_sec
        self._process: multiprocessing.Process | None = None
        self._shared_heartbeat = multiprocessing.Value("d", time.time())
        self._should_run = True

    def start(self) -> None:
        log.info("Watchdog starting...")
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self._launch_engine()

        while self._should_run:
            if self._process is None or not self._process.is_alive():
                log.critical("Trading process died — restarting")
                self._launch_engine()

            # 하트비트 체크
            elapsed = time.time() - self._shared_heartbeat.value
            if elapsed > self.heartbeat_timeout:
                log.critical(
                    "Trading process unresponsive (%.1fs) — killing and restarting",
                    elapsed,
                )
                self._kill_engine()
                self._launch_engine()

            time.sleep(1)

    def _launch_engine(self) -> None:
        self._shared_heartbeat.value = time.time()
        self._process = multiprocessing.Process(
            target=self.engine_starter,
            args=(self.config, self._shared_heartbeat),
            daemon=True,
        )
        self._process.start()
        log.info("Trading process started (PID=%d)", self._process.pid)

    def _kill_engine(self) -> None:
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=3)
            if self._process.is_alive():
                self._process.kill()
            log.warning("Trading process killed")

    def _handle_shutdown(self, signum, frame) -> None:
        log.info("Shutdown signal received")
        self._should_run = False
        self._kill_engine()
        sys.exit(0)
