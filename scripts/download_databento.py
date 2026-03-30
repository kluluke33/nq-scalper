"""
Databento에서 NQ 과거 틱 데이터를 다운로드.

사전 준비:
  1. https://databento.com 에서 계정 생성
  2. API 키 발급
  3. pip install databento

사용법:
    # 환경변수로 API 키 설정
    export DATABENTO_API_KEY="your-api-key"

    # 최근 1년 다운로드
    python scripts/download_databento.py --months 12

    # 특정 기간
    python scripts/download_databento.py --start 2025-01-01 --end 2025-12-31

    # 출력 디렉토리 지정
    python scripts/download_databento.py --months 12 --output data/raw
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

ET = timezone(timedelta(hours=-4))

# NQ 거래 시간 (ET) — 우리는 09:30~10:30만 사용
SESSION_START_HOUR = 9
SESSION_START_MIN = 30
SESSION_END_HOUR = 10
SESSION_END_MIN = 30


def download_from_databento(
    api_key: str,
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw",
) -> None:
    """Databento API로 NQ 틱 데이터를 세션별 CSV로 저장."""
    try:
        import databento as db
    except ImportError:
        log.error("databento 패키지가 필요합니다: pip install databento")
        sys.exit(1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    client = db.Historical(key=api_key)

    log.info("Requesting NQ tick data: %s ~ %s", start_date, end_date)

    # Databento에서 데이터 요청
    # CME Globex의 NQ 연속 계약 (front month)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",          # CME Globex
        symbols=["NQ.c.0"],            # NQ 선물 연속 계약
        schema="trades",               # 체결 틱
        start=start_date,
        end=end_date,
        stype_in="continuous",
    )

    # DataFrame 변환
    df = data.to_df()
    log.info("Total ticks received: %d", len(df))

    if df.empty:
        log.warning("No data received")
        return

    # 타임스탬프를 ET로 변환
    df.index = df.index.tz_convert(ET)

    # 세션별 분할 (09:30~10:30 ET만 추출)
    df["date"] = df.index.date
    df["time"] = df.index.time

    from datetime import time as dt_time
    session_start = dt_time(SESSION_START_HOUR, SESSION_START_MIN)
    session_end = dt_time(SESSION_END_HOUR, SESSION_END_MIN)

    session_mask = (df["time"] >= session_start) & (df["time"] <= session_end)
    df_session = df[session_mask]

    log.info("Ticks in trading window (09:30~10:30 ET): %d", len(df_session))

    # 날짜별로 CSV 저장
    dates = df_session["date"].unique()
    for d in sorted(dates):
        date_str = d.strftime("%Y%m%d")
        out_file = out / f"ticks_{date_str}.csv"

        if out_file.exists():
            log.info("Skip %s — already exists", date_str)
            continue

        day_df = df_session[df_session["date"] == d]

        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "price", "size", "side"])
            for ts, row in day_df.iterrows():
                # Databento의 side: 'A'=ask(buy), 'B'=bid(sell)
                side_val = row.get("side", "")
                if side_val == "A":
                    side = 1  # buy (aggressor hit the ask)
                elif side_val == "B":
                    side = -1  # sell (aggressor hit the bid)
                else:
                    side = 0

                writer.writerow([
                    ts.isoformat(),
                    row["price"],
                    int(row["size"]),
                    side,
                ])

        log.info("%s: %d ticks", date_str, len(day_df))

    log.info("Download complete — %d sessions saved to %s", len(dates), output_dir)


def main():
    parser = argparse.ArgumentParser(description="Download NQ tick data from Databento")
    parser.add_argument("--months", type=int, default=12, help="Number of months to download")
    parser.add_argument("--start", type=str, default="", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--api-key", default="", help="Databento API key (or set DATABENTO_API_KEY)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        log.error(
            "API 키가 필요합니다.\n"
            "  --api-key YOUR_KEY  또는\n"
            "  export DATABENTO_API_KEY=YOUR_KEY"
        )
        sys.exit(1)

    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        end = date.today()
        start = end - timedelta(days=args.months * 30)
        start_date = start.isoformat()
        end_date = end.isoformat()

    download_from_databento(api_key, start_date, end_date, args.output)


if __name__ == "__main__":
    main()
