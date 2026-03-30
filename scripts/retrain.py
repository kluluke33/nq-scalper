"""
모델 재학습 스크립트.

기존 Databento 데이터 + 매일 축적된 IBKR 틱 데이터를 합쳐서
모델을 재학습한다.

실행 시점:
  - 매주 주말 (cron)
  - 성과 하락 감지 시 (수동)

사용법:
    python scripts/retrain.py
    python scripts/retrain.py --tick-count 500 --tp 15 --sl 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_bars import ticks_to_bars
from scripts.train_model import load_sessions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def rebuild_bars(tick_dir: str, bars_dir: str, tick_count: int) -> None:
    """신규 틱 파일이 있으면 바 CSV로 변환."""
    import csv

    tick_path = Path(tick_dir)
    bars_path = Path(bars_dir)
    bars_path.mkdir(parents=True, exist_ok=True)

    new_count = 0
    for tick_file in sorted(tick_path.glob("ticks_*.csv")):
        date_str = tick_file.stem.replace("ticks_", "")
        bar_file = bars_path / f"bars_{date_str}.csv"

        if bar_file.exists():
            continue

        bars = ticks_to_bars(tick_file, tick_count)
        if not bars:
            continue

        with open(bar_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=bars[0].keys())
            writer.writeheader()
            writer.writerows(bars)

        new_count += 1
        log.info("New bars: %s (%d bars)", date_str, len(bars))

    if new_count:
        log.info("Built %d new bar files", new_count)
    else:
        log.info("No new tick files to process")


def main():
    parser = argparse.ArgumentParser(description="Retrain model with accumulated data")
    parser.add_argument("--tick-dir", default="data/raw")
    parser.add_argument("--bars-dir", default="data/bars")
    parser.add_argument("--tick-count", type=int, default=500)
    parser.add_argument("--tp", type=float, default=15.0)
    parser.add_argument("--sl", type=float, default=10.0)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--name", default="lgbm_nq")
    args = parser.parse_args()

    # Step 1: 신규 틱 → 바 변환
    log.info("Step 1: Building bars from new tick data...")
    rebuild_bars(args.tick_dir, args.bars_dir, args.tick_count)

    # Step 2: 전체 세션 로드
    log.info("Step 2: Loading all sessions...")
    sessions, prev_day_closes = load_sessions(args.bars_dir)

    if len(sessions) < 20:
        log.error("Not enough sessions (%d) — need at least 20", len(sessions))
        sys.exit(1)

    # Step 3: 학습
    from src.model.train import ModelTrainer

    trainer = ModelTrainer({
        "tp_points": args.tp,
        "sl_points": args.sl,
    })

    log.info("Step 3: Building features and labels...")
    features, labels = trainer.prepare_dataset(sessions, prev_day_closes)

    if args.tune:
        from scripts.train_model import tune_hyperparameters
        log.info("Step 3.5: Hyperparameter tuning...")
        best_params = tune_hyperparameters(trainer, features, labels)
        trainer.lgb_params = best_params

    log.info("Step 4: Training model...")
    model = trainer.train(features, labels)

    # Step 4: 저장
    model_path = trainer.save_model(model, name=args.name)

    log.info("=" * 60)
    log.info("Retrain complete!")
    log.info("  Sessions used: %d", len(sessions))
    log.info("  Samples:       %d", len(labels))
    log.info("  Model:         %s", model_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
