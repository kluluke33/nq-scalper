"""
세션 관리자.

거래 시간:
  - 미국장 개장 (09:30 ET) + 5분 = 09:35 ET 부터 거래 시작
  - 개장 후 1시간 = 10:30 ET 에 거래 종료
  - 10:30 도달 시 모든 포지션 청산, 신규 진입 차단

경제지표 블랙아웃:
  - 지표 발표 5분 전 ~ 5분 후 (총 10분) 거래 금지
  - 블랙아웃 진입 시 포지션이 있으면 즉시 청산
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone
from enum import Enum, auto

log = logging.getLogger(__name__)

# 미국 동부시간 (ET)
ET = timezone(timedelta(hours=-4))   # EDT (서머타임)
ET_STANDARD = timezone(timedelta(hours=-5))  # EST


class SessionPhase(Enum):
    PRE_MARKET = auto()      # 09:30 이전 — 시스템 웜업
    WARMUP = auto()          # 09:30~09:35 — 데이터 수집만, 거래 안 함
    ACTIVE = auto()          # 09:35~10:30 — 거래 가능
    BLACKOUT = auto()        # 지표 발표 전후 — 거래 금지
    DONE = auto()            # 10:30 이후 — 오늘 거래 종료


class EconomicEvent:
    """경제지표 발표 이벤트."""

    __slots__ = ("name", "release_time", "impact")

    def __init__(self, name: str, release_time: datetime, impact: str = "high"):
        self.name = name
        self.release_time = release_time
        self.impact = impact  # high, medium, low

    def blackout_start(self, buffer_minutes: int = 5) -> datetime:
        return self.release_time - timedelta(minutes=buffer_minutes)

    def blackout_end(self, buffer_minutes: int = 5) -> datetime:
        return self.release_time + timedelta(minutes=buffer_minutes)


class SessionManager:

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        session = cfg.get("session", {})

        self.market_open = self._parse_time(session.get("market_open", "09:30"))
        self.trade_start_delay_min = session.get("trade_start_delay_min", 5)
        self.trade_duration_min = session.get("trade_duration_min", 60)
        self.blackout_buffer_min = session.get("blackout_buffer_min", 5)
        self.flatten_before_end_sec = session.get("flatten_before_end_sec", 30)

        # 계산된 시간
        self._trade_start = self._add_minutes(self.market_open, self.trade_start_delay_min)
        self._trade_end = self._add_minutes(self.market_open, self.trade_duration_min)
        self._flatten_time = self._add_minutes(
            self.market_open,
            self.trade_duration_min - (self.flatten_before_end_sec / 60),
        )

        # 오늘의 경제지표 이벤트
        self._events: list[EconomicEvent] = []

        # 상태
        self._phase = SessionPhase.PRE_MARKET
        self._prev_phase = SessionPhase.PRE_MARKET
        self._blackout_reason = ""

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    @property
    def phase_changed(self) -> bool:
        return self._phase != self._prev_phase

    @property
    def can_trade(self) -> bool:
        return self._phase == SessionPhase.ACTIVE

    @property
    def should_flatten(self) -> bool:
        """포지션 청산이 필요한 상태인가."""
        return self._phase in (SessionPhase.DONE, SessionPhase.BLACKOUT)

    @property
    def is_done(self) -> bool:
        return self._phase == SessionPhase.DONE

    def set_economic_events(self, events: list[EconomicEvent]) -> None:
        """오늘의 경제지표 발표 일정을 설정."""
        # 거래 시간 내 이벤트만 필터링
        relevant = []
        for ev in events:
            ev_time = ev.release_time.time()
            if self.market_open <= ev_time <= self._trade_end:
                relevant.append(ev)
                log.info(
                    "Economic event registered: %s at %s (blackout %s~%s)",
                    ev.name,
                    ev.release_time.strftime("%H:%M"),
                    ev.blackout_start(self.blackout_buffer_min).strftime("%H:%M"),
                    ev.blackout_end(self.blackout_buffer_min).strftime("%H:%M"),
                )
        self._events = relevant

        if not relevant:
            log.info("No economic events during trading window")

    def update(self, now: datetime | None = None) -> SessionPhase:
        """현재 시각을 기준으로 세션 상태를 갱신."""
        if now is None:
            now = datetime.now(ET)

        current_time = now.time()
        self._prev_phase = self._phase

        # 10:30 이후 → 종료
        if current_time >= self._trade_end:
            self._phase = SessionPhase.DONE
            if self.phase_changed:
                log.info("SESSION: Trading window closed (10:30 ET)")
            return self._phase

        # 09:30 이전 → 대기
        if current_time < self.market_open:
            self._phase = SessionPhase.PRE_MARKET
            return self._phase

        # 09:30~09:35 → 웜업 (데이터 수집만)
        if current_time < self._trade_start:
            self._phase = SessionPhase.WARMUP
            if self.phase_changed:
                log.info("SESSION: Warmup phase — collecting data, no trades")
            return self._phase

        # 09:35~10:30 → 블랙아웃 체크 후 ACTIVE
        for ev in self._events:
            blackout_start = ev.blackout_start(self.blackout_buffer_min).time()
            blackout_end = ev.blackout_end(self.blackout_buffer_min).time()
            if blackout_start <= current_time <= blackout_end:
                self._phase = SessionPhase.BLACKOUT
                self._blackout_reason = ev.name
                if self.phase_changed:
                    log.warning(
                        "SESSION: Blackout for '%s' (until %s)",
                        ev.name,
                        ev.blackout_end(self.blackout_buffer_min).strftime("%H:%M:%S"),
                    )
                return self._phase

        # 청산 준비 시간 (10:29:30~10:30)
        if current_time >= self._flatten_time:
            self._phase = SessionPhase.DONE
            if self.phase_changed:
                log.info("SESSION: Flatten time — closing positions")
            return self._phase

        self._phase = SessionPhase.ACTIVE
        if self.phase_changed:
            log.info("SESSION: Active trading window")
        return self._phase

    def time_remaining_sec(self, now: datetime | None = None) -> float:
        """거래 종료까지 남은 초."""
        if now is None:
            now = datetime.now(ET)
        end_dt = now.replace(
            hour=self._trade_end.hour,
            minute=self._trade_end.minute,
            second=0,
            microsecond=0,
        )
        remaining = (end_dt - now).total_seconds()
        return max(0.0, remaining)

    def summary(self) -> dict:
        return {
            "phase": self._phase.name,
            "trade_window": f"{self._trade_start.strftime('%H:%M')}~{self._trade_end.strftime('%H:%M')} ET",
            "economic_events": [
                {"name": ev.name, "time": ev.release_time.strftime("%H:%M")}
                for ev in self._events
            ],
            "blackout_reason": self._blackout_reason if self._phase == SessionPhase.BLACKOUT else "",
        }

    @staticmethod
    def _parse_time(s: str) -> time:
        parts = s.split(":")
        return time(int(parts[0]), int(parts[1]))

    @staticmethod
    def _add_minutes(t: time, minutes: float) -> time:
        total_min = t.hour * 60 + t.minute + minutes
        return time(int(total_min // 60), int(total_min % 60))
