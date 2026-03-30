"""
모델 학습 파이프라인.

흐름:
  1. 과거 틱 데이터 로드 → 틱 차트 바 생성
  2. 세션별(일별) 분할 → 피처 생성 + 라벨링
  3. 시간순 분할 (Walk-Forward) → 학습/검증
  4. LightGBM 학습 → 모델 저장
  5. 성과 리포트 출력

핵심 원칙:
  - 미래 정보 유출(look-ahead bias) 방지: 시간순 분할만 사용
  - 세션 경계 존중: 다른 날의 데이터가 라벨링에 혼입되지 않도록
  - 재현성: 시드 고정, 파라미터 기록
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score,
)

from src.model.features import FEATURE_NAMES, build_features_from_bars
from src.model.labeler import label_bars_bidirectional

log = logging.getLogger(__name__)


class ModelTrainer:

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.tp_points = cfg.get("tp_points", 15.0)
        self.sl_points = cfg.get("sl_points", 10.0)
        self.max_bars_ahead = cfg.get("max_bars_ahead", 50)
        self.test_ratio = cfg.get("test_ratio", 0.2)
        self.seed = cfg.get("seed", 42)
        self.model_dir = Path(cfg.get("model_dir", "src/model/models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # LightGBM 하이퍼파라미터
        self.lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": 6,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "seed": self.seed,
            "verbose": -1,
        }
        # 사용자 오버라이드
        if "lgb_params" in cfg:
            self.lgb_params.update(cfg["lgb_params"])

    def prepare_dataset(
        self,
        sessions: list[pd.DataFrame],
        prev_day_closes: list[float] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        세션(일)별 바 데이터를 피처+라벨로 변환.

        Args:
            sessions: 일별 바 DataFrame 리스트
                      각 DataFrame: open, high, low, close, tick_count,
                                    buy_ticks, sell_ticks, duration_ms, timestamp_ns
            prev_day_closes: 각 세션의 전일 종가 리스트

        Returns:
            (features_df, labels_series)
        """
        all_features = []
        all_labels = []

        if prev_day_closes is None:
            prev_day_closes = [0.0] * len(sessions)

        for i, (session_df, prev_close) in enumerate(zip(sessions, prev_day_closes)):
            if len(session_df) < 30:
                log.warning("Session %d: too few bars (%d), skipping", i, len(session_df))
                continue

            # 피처 생성 (세션 단위)
            features = build_features_from_bars(session_df, prev_close)

            # 라벨링 (세션 단위 — 다른 날 데이터 혼입 방지)
            labels = label_bars_bidirectional(
                session_df,
                tp_points=self.tp_points,
                sl_points=self.sl_points,
                max_bars_ahead=self.max_bars_ahead,
            )

            all_features.append(features)
            all_labels.append(labels)

        features_df = pd.concat(all_features, ignore_index=True)
        labels_series = pd.concat(all_labels, ignore_index=True)

        # NaN 라벨 제거 (불확실 샘플)
        valid_mask = labels_series.notna()
        features_df = features_df[valid_mask].reset_index(drop=True)
        labels_series = labels_series[valid_mask].reset_index(drop=True)

        log.info(
            "Dataset: %d samples (%.1f%% LONG, %.1f%% SHORT)",
            len(labels_series),
            labels_series.mean() * 100,
            (1 - labels_series.mean()) * 100,
        )

        return features_df, labels_series

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> lgb.Booster:
        """
        시간순 분할 후 LightGBM 학습.

        Walk-Forward 방식:
          앞쪽 80% → 학습
          뒤쪽 20% → 검증 (미래 데이터)
          ※ 랜덤 분할 금지 — 미래 정보 유출 방지
        """
        n = len(features)
        split_idx = int(n * (1 - self.test_ratio))

        X_train = features.iloc[:split_idx]
        y_train = labels.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_test = labels.iloc[split_idx:]

        log.info("Train: %d samples | Test: %d samples", len(X_train), len(X_test))
        log.info(
            "Train label dist: %.1f%% LONG | Test label dist: %.1f%% LONG",
            y_train.mean() * 100, y_test.mean() * 100,
        )

        # LightGBM 데이터셋
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # 학습
        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=50),
        ]

        model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            valid_names=["train", "test"],
            callbacks=callbacks,
        )

        # 검증 성과
        self._evaluate(model, X_test, y_test)

        return model

    def save_model(self, model: lgb.Booster, name: str = "lgbm_nq") -> str:
        """모델과 메타데이터를 저장."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{name}_{timestamp}"

        # 모델 파일
        model_path = self.model_dir / f"{model_name}.pkl"
        model.save_model(str(model_path))

        # 메타데이터
        meta = {
            "name": model_name,
            "created": timestamp,
            "feature_count": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "params": self.lgb_params,
            "tp_points": self.tp_points,
            "sl_points": self.sl_points,
            "best_iteration": model.best_iteration,
        }
        meta_path = self.model_dir / f"{model_name}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # 심볼릭 링크 — 최신 모델
        latest_link = self.model_dir / f"{name}_latest.pkl"
        if latest_link.exists():
            latest_link.unlink()
        # 상대경로로 심볼릭 링크
        latest_link.symlink_to(model_path.name)

        log.info("Model saved: %s", model_path)
        log.info("Latest link: %s → %s", latest_link, model_path.name)

        return str(model_path)

    def _evaluate(self, model: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """검증 데이터에 대한 성과 평가."""
        y_prob = model.predict(X_test)
        y_pred = (y_prob > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        logloss = log_loss(y_test, y_prob)

        log.info("=" * 50)
        log.info("Model Evaluation (Test Set)")
        log.info("=" * 50)
        log.info("Accuracy:  %.4f", acc)
        log.info("AUC-ROC:   %.4f", auc)
        log.info("Log Loss:  %.4f", logloss)
        log.info("")
        log.info(classification_report(y_test, y_pred, target_names=["SHORT", "LONG"]))

        # 피처 중요도
        importance = model.feature_importance(importance_type="gain")
        importance_df = pd.DataFrame({
            "feature": FEATURE_NAMES,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        log.info("Top 10 Features:")
        for _, row in importance_df.head(10).iterrows():
            log.info("  %-25s %.1f", row["feature"], row["importance"])

        # 확신도별 정확도
        log.info("")
        log.info("Accuracy by Confidence Level:")
        for threshold in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            high_conf_mask = (y_prob > threshold) | (y_prob < (1 - threshold))
            if high_conf_mask.sum() > 0:
                acc_high = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])
                pct = high_conf_mask.mean() * 100
                log.info(
                    "  conf > %.2f: accuracy=%.4f (%.1f%% of samples)",
                    threshold, acc_high, pct,
                )

        return {
            "accuracy": acc,
            "auc": auc,
            "log_loss": logloss,
            "feature_importance": importance_df.to_dict("records"),
        }
