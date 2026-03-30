"""
모델 학습 실행 스크립트.

흐름:
  1. 세션별 바 CSV 로드
  2. 피처 생성 + 라벨링
  3. 시간순 분할 → LightGBM 학습
  4. 모델 저장 + 성과 리포트

사용법:
    python scripts/train_model.py
    python scripts/train_model.py --bars-dir data/bars --tp 15 --sl 10
    python scripts/train_model.py --tune   # 하이퍼파라미터 튜닝
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.train import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def load_sessions(bars_dir: str) -> tuple[list[pd.DataFrame], list[float]]:
    """세션별 바 CSV를 로드."""
    bar_files = sorted(Path(bars_dir).glob("bars_*.csv"))
    if not bar_files:
        log.error("No bar files found in %s", bars_dir)
        sys.exit(1)

    sessions = []
    prev_day_closes = []
    prev_close = 0.0

    for f in bar_files:
        df = pd.read_csv(f)
        if len(df) < 10:
            continue

        prev_day_closes.append(prev_close)
        prev_close = df["close"].iloc[-1]  # 이 세션의 마지막 종가 = 다음 세션의 전일 종가
        sessions.append(df)

    log.info("Loaded %d sessions from %s", len(sessions), bars_dir)
    return sessions, prev_day_closes


def tune_hyperparameters(trainer: ModelTrainer, features: pd.DataFrame, labels: pd.Series) -> dict:
    """간이 하이퍼파라미터 튜닝."""
    import lightgbm as lgb

    n = len(features)
    split_idx = int(n * 0.8)

    X_train = features.iloc[:split_idx]
    y_train = labels.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_test = labels.iloc[split_idx:]

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 탐색할 파라미터 조합
    param_grid = [
        {"num_leaves": 15, "max_depth": 4, "learning_rate": 0.03},
        {"num_leaves": 31, "max_depth": 6, "learning_rate": 0.05},
        {"num_leaves": 63, "max_depth": 8, "learning_rate": 0.05},
        {"num_leaves": 31, "max_depth": 6, "learning_rate": 0.1},
        {"num_leaves": 15, "max_depth": 5, "learning_rate": 0.03, "min_child_samples": 50},
        {"num_leaves": 31, "max_depth": 6, "learning_rate": 0.05, "feature_fraction": 0.6},
    ]

    best_score = float("inf")
    best_params = {}

    for i, params in enumerate(param_grid):
        trial_params = {**trainer.lgb_params, **params}

        model = lgb.train(
            trial_params,
            train_data,
            num_boost_round=500,
            valid_sets=[test_data],
            valid_names=["test"],
            callbacks=[
                lgb.early_stopping(30),
                lgb.log_evaluation(0),  # 조용히
            ],
        )

        score = model.best_score["test"]["binary_logloss"]
        log.info(
            "Trial %d/%d: logloss=%.4f (leaves=%d, depth=%d, lr=%.3f)",
            i + 1, len(param_grid), score,
            params.get("num_leaves", "?"),
            params.get("max_depth", "?"),
            params.get("learning_rate", "?"),
        )

        if score < best_score:
            best_score = score
            best_params = trial_params

    log.info("Best params (logloss=%.4f): %s", best_score, best_params)
    return best_params


def main():
    parser = argparse.ArgumentParser(description="Train NQ scalping model")
    parser.add_argument("--bars-dir", default="data/bars")
    parser.add_argument("--tp", type=float, default=15.0, help="Take profit points")
    parser.add_argument("--sl", type=float, default=10.0, help="Stop loss points")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--name", default="lgbm_nq", help="Model name prefix")
    args = parser.parse_args()

    # 세션 로드
    sessions, prev_day_closes = load_sessions(args.bars_dir)

    # 트레이너 초기화
    trainer = ModelTrainer({
        "tp_points": args.tp,
        "sl_points": args.sl,
    })

    # 데이터셋 생성
    log.info("Building features and labels...")
    features, labels = trainer.prepare_dataset(sessions, prev_day_closes)

    log.info("=" * 60)
    log.info("Dataset Summary")
    log.info("  Total samples:  %d", len(labels))
    log.info("  LONG labels:    %d (%.1f%%)", (labels == 1).sum(), labels.mean() * 100)
    log.info("  SHORT labels:   %d (%.1f%%)", (labels == 0).sum(), (1 - labels.mean()) * 100)
    log.info("  Features:       %d", features.shape[1])
    log.info("=" * 60)

    # 하이퍼파라미터 튜닝 (선택)
    if args.tune:
        log.info("Running hyperparameter tuning...")
        best_params = tune_hyperparameters(trainer, features, labels)
        trainer.lgb_params = best_params

    # 학습
    log.info("Training model...")
    model = trainer.train(features, labels)

    # 저장
    model_path = trainer.save_model(model, name=args.name)
    log.info("=" * 60)
    log.info("Training complete!")
    log.info("Model: %s", model_path)
    log.info("")
    log.info("To use this model, update config/settings.yaml:")
    log.info("  strategy.ai.model_path: %s", model_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
