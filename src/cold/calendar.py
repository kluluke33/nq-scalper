"""
경제지표 캘린더.

당일 주요 경제지표 발표 일정을 가져와서 SessionManager에 제공.

데이터 소스:
  1. Investing.com 무료 캘린더 (웹 스크래핑)
  2. 로컬 수동 설정 (config/economic_events.yaml)
  3. FRED API (선택)

거래 시간이 09:35~10:30 ET 이므로,
이 시간대에 발표되는 주요 지표만 필터링.

09:35~10:30 ET에 자주 발표되는 주요 지표:
  - 10:00 ET: ISM 제조업/서비스업, 소비자신뢰지수, 신규주택판매, JOLTS
  - 10:30 ET: EIA 원유재고 (수요일)
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import yaml

from src.hot.session import ET, EconomicEvent

log = logging.getLogger(__name__)


def load_events_from_file(
    file_path: str = "config/economic_events.yaml",
    date: datetime | None = None,
) -> list[EconomicEvent]:
    """
    로컬 파일에서 경제지표 일정을 로드.

    파일 형식 (config/economic_events.yaml):
        events:
          - date: "2024-06-03"
            time: "10:00"
            name: "ISM Manufacturing PMI"
            impact: high
          - date: "2024-06-05"
            time: "10:30"
            name: "EIA Crude Oil Inventories"
            impact: high
    """
    path = Path(file_path)
    if not path.exists():
        log.info("No economic events file found: %s", file_path)
        return []

    if date is None:
        date = datetime.now(ET)

    today_str = date.strftime("%Y-%m-%d")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    events = []
    for entry in data.get("events", []):
        if entry.get("date") != today_str:
            continue

        time_parts = entry["time"].split(":")
        release_dt = date.replace(
            hour=int(time_parts[0]),
            minute=int(time_parts[1]),
            second=0,
            microsecond=0,
        )

        events.append(EconomicEvent(
            name=entry["name"],
            release_time=release_dt,
            impact=entry.get("impact", "high"),
        ))

    log.info("Loaded %d economic events for %s", len(events), today_str)
    return events


async def fetch_events_from_web(date: datetime | None = None) -> list[EconomicEvent]:
    """
    웹에서 당일 경제지표 일정을 가져옴.
    Investing.com 캘린더를 파싱.

    주의: 실전에서는 API 키 기반 서비스 권장 (예: Trading Economics API).
    """
    try:
        import aiohttp
        from bs4 import BeautifulSoup
    except ImportError:
        log.warning("aiohttp/beautifulsoup4 not installed — skipping web calendar")
        return []

    if date is None:
        date = datetime.now(ET)

    date_str = date.strftime("%Y/%m/%d")
    url = f"https://www.investing.com/economic-calendar/"

    events = []
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"User-Agent": "Mozilla/5.0"}
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    log.warning("Failed to fetch economic calendar: HTTP %d", resp.status)
                    return []
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")
        # 파싱 로직은 사이트 구조 변경 시 업데이트 필요
        # 여기서는 구조만 제공, 실제 파싱은 사이트별로 구현
        log.info("Fetched economic calendar from web")

    except Exception:
        log.exception("Failed to fetch economic calendar")

    return events


def create_manual_events(entries: list[dict]) -> list[EconomicEvent]:
    """
    코드에서 직접 이벤트를 생성.

    사용법:
        events = create_manual_events([
            {"time": "10:00", "name": "ISM Manufacturing PMI"},
            {"time": "10:30", "name": "EIA Crude Oil Inventories"},
        ])
    """
    today = datetime.now(ET)
    events = []
    for entry in entries:
        time_parts = entry["time"].split(":")
        release_dt = today.replace(
            hour=int(time_parts[0]),
            minute=int(time_parts[1]),
            second=0,
            microsecond=0,
        )
        events.append(EconomicEvent(
            name=entry["name"],
            release_time=release_dt,
            impact=entry.get("impact", "high"),
        ))
    return events
