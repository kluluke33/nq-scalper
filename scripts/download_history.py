"""
IBKR에서 과거 틱 데이터를 다운로드하여 세션별 CSV로 저장.

다운로드 범위: 최근 N일의 09:30~10:30 ET 구간
저장 형식:     data/raw/ticks_YYYYMMDD.csv

사용법:
    python scripts/download_history.py --days 120
    python scripts/download_history.py --days 60 --output data/raw
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

ET = timezone(timedelta(hours=-4))


async def download_ticks(
    days: int = 120,
    output_dir: str = "data/raw",
    host: str = "127.0.0.1",
    port: int = 7497,
) -> None:
    """IBKR에서 과거 틱 데이터를 다운로드."""
    from ib_insync import IB, Future, util

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ib = IB()
    await ib.connectAsync(host=host, port=port, clientId=99)

    contract = Future(symbol="NQ", exchange="CME")
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        log.error("Failed to qualify NQ contract")
        return
    contract = qualified[0]

    today = datetime.now(ET).date()

    for day_offset in range(1, days + 1):
        date = today - timedelta(days=day_offset)
        # 주말 건너뛰기
        if date.weekday() >= 5:
            continue

        date_str = date.strftime("%Y%m%d")
        out_file = out / f"ticks_{date_str}.csv"

        if out_file.exists():
            log.info("Skip %s — already exists", date_str)
            continue

        # 09:30~10:30 ET
        start_dt = datetime(date.year, date.month, date.day, 9, 30, tzinfo=ET)
        end_dt = datetime(date.year, date.month, date.day, 10, 30, tzinfo=ET)
        end_str = end_dt.strftime("%Y%m%d %H:%M:%S US/Eastern")

        try:
            ticks = await ib.reqHistoricalTicksAsync(
                contract,
                startDateTime="",
                endDateTime=end_str,
                numberOfTicks=1000,
                whatToShow="TRADES",
                useRth=False,
            )

            if not ticks:
                log.warning("No ticks for %s", date_str)
                continue

            # CSV 저장
            with open(out_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "price", "size", "exchange"])
                for t in ticks:
                    writer.writerow([
                        t.time.isoformat(),
                        t.price,
                        t.size,
                        getattr(t, "exchange", ""),
                    ])

            log.info("Downloaded %s: %d ticks", date_str, len(ticks))

            # API 속도 제한 준수
            await asyncio.sleep(1)

        except Exception:
            log.exception("Failed to download %s", date_str)

    ib.disconnect()
    log.info("Download complete — files in %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Download NQ tick history from IBKR")
    parser.add_argument("--days", type=int, default=120, help="Number of days to download")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    args = parser.parse_args()

    asyncio.run(download_ticks(
        days=args.days,
        output_dir=args.output,
        host=args.host,
        port=args.port,
    ))


if __name__ == "__main__":
    main()
