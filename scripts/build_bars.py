"""
과거 틱 CSV → 세션별 틱 차트 바 CSV 변환.

입력:  data/raw/ticks_YYYYMMDD.csv
출력:  data/bars/bars_YYYYMMDD.csv

사용법:
    python scripts/build_bars.py                       # 기본 500틱
    python scripts/build_bars.py --tick-count 300
    python scripts/build_bars.py --input data/raw --output data/bars
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def ticks_to_bars(tick_file: Path, tick_count: int) -> list[dict]:
    """틱 CSV 파일을 읽어 N-tick 바를 생성."""
    ticks = []
    with open(tick_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticks.append({
                "timestamp": row["timestamp"],
                "price": float(row["price"]),
                "size": int(row["size"]),
            })

    if not ticks:
        return []

    bars = []
    for i in range(0, len(ticks) - tick_count + 1, tick_count):
        chunk = ticks[i:i + tick_count]
        prices = [t["price"] for t in chunk]

        # 매수/매도 추정 (틱 방향: 직전 가격 대비)
        buy_ticks = 0
        sell_ticks = 0
        for j in range(1, len(chunk)):
            if chunk[j]["price"] > chunk[j - 1]["price"]:
                buy_ticks += 1
            elif chunk[j]["price"] < chunk[j - 1]["price"]:
                sell_ticks += 1
            else:
                # 동가: 이전 방향 유지 (간이 처리)
                buy_ticks += 1

        # 첫 틱은 매수로 가정
        buy_ticks += 1

        t0 = datetime.fromisoformat(chunk[0]["timestamp"])
        t1 = datetime.fromisoformat(chunk[-1]["timestamp"])
        duration_ms = (t1 - t0).total_seconds() * 1000

        bars.append({
            "timestamp": chunk[0]["timestamp"],
            "timestamp_ns": int(t0.timestamp() * 1e9),
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "tick_count": len(chunk),
            "buy_ticks": buy_ticks,
            "sell_ticks": sell_ticks,
            "duration_ms": duration_ms,
        })

    return bars


def main():
    parser = argparse.ArgumentParser(description="Convert tick CSV to bar CSV")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/bars")
    parser.add_argument("--tick-count", type=int, default=500)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tick_files = sorted(input_dir.glob("ticks_*.csv"))
    if not tick_files:
        log.error("No tick files found in %s", input_dir)
        return

    total_bars = 0
    for tick_file in tick_files:
        date_str = tick_file.stem.replace("ticks_", "")
        out_file = output_dir / f"bars_{date_str}.csv"

        if out_file.exists():
            log.info("Skip %s — already exists", date_str)
            continue

        bars = ticks_to_bars(tick_file, args.tick_count)
        if not bars:
            log.warning("No bars generated for %s", date_str)
            continue

        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=bars[0].keys())
            writer.writeheader()
            writer.writerows(bars)

        total_bars += len(bars)
        log.info("%s: %d bars", date_str, len(bars))

    log.info("Total: %d bars from %d sessions", total_bars, len(tick_files))


if __name__ == "__main__":
    main()
