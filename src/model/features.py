"""
피처 정의 및 과거 데이터에서의 일괄 피처 생성.

두 가지 사용처:
  1. 실시간: IncrementalFeatures (ai_strategy.py) — 매 바 O(1) 업데이트
  2. 학습:  이 모듈의 build_features() — 과거 데이터에서 일괄 생성

두 곳의 피처 정의는 반드시 동일해야 함.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.ai_strategy import IncrementalFeatures


FEATURE_NAMES = IncrementalFeatures.feature_names()
FEATURE_COUNT = IncrementalFeatures.FEATURE_COUNT


def build_features_from_bars(bars_df: pd.DataFrame, prev_day_close: float = 0.0) -> pd.DataFrame:
    """
    바 DataFrame에서 피처를 일괄 생성.

    bars_df 컬럼:
      open, high, low, close, tick_count, buy_ticks, sell_ticks,
      duration_ms, timestamp_ns

    반환: 피처 30개 컬럼이 추가된 DataFrame
    """
    df = bars_df.copy()
    n = len(df)

    # 기본 가격
    df["change"] = df["close"].diff().fillna(0)
    df["ema_fast"] = df["close"].ewm(span=8, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
    df["macd"] = df["ema_fast"] - df["ema_slow"]

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    df["rsi"] = 100 - (100 / (1 + rs))

    # 바 구조
    df["bar_range"] = df["high"] - df["low"]
    df["bar_body"] = (df["close"] - df["open"]).abs()
    bar_range_safe = df["bar_range"].replace(0, np.nan)
    df["close_position"] = ((df["close"] - df["low"]) / bar_range_safe).fillna(0.5)

    # 주문 흐름
    df["delta"] = df["buy_ticks"] - df["sell_ticks"]
    df["buy_ratio"] = df["buy_ticks"] / df["tick_count"].replace(0, 1)
    df["cum_delta"] = df["delta"].cumsum()

    # 일중 컨텍스트
    df["day_high"] = df["high"].cummax()
    df["day_low"] = df["low"].cummin()
    day_range = df["day_high"] - df["day_low"]
    df["day_position"] = ((df["close"] - df["day_low"]) / day_range.replace(0, np.nan)).fillna(0.5)
    df["dist_ema_fast"] = df["close"] - df["ema_fast"]
    df["dist_ema_slow"] = df["close"] - df["ema_slow"]

    # 시퀀스 대체
    df["momentum_3bar"] = df["close"].diff(3).fillna(0)
    df["momentum_5bar"] = df["close"].diff(5).fillna(0)
    df["delta_sum_3bar"] = df["delta"].rolling(3, min_periods=1).sum()
    df["delta_sum_5bar"] = df["delta"].rolling(5, min_periods=1).sum()
    df["bullish_ratio_3bar"] = (df["close"] >= df["open"]).astype(float).rolling(3, min_periods=1).mean()
    avg_range_5 = df["bar_range"].rolling(5, min_periods=1).mean()
    df["range_ratio"] = (df["bar_range"] / avg_range_5.replace(0, np.nan)).fillna(1.0)

    # 개장 직후 특화
    first_open = df["open"].iloc[0] if n > 0 else 0
    df["gap"] = first_open - prev_day_close if prev_day_close > 0 else 0.0

    gap_val = df["gap"].iloc[0] if n > 0 else 0
    if abs(gap_val) > 0.25:
        if gap_val > 0:
            df["gap_fill_ratio"] = ((first_open - df["day_low"]) / gap_val).clip(0, 1)
        else:
            df["gap_fill_ratio"] = ((df["day_high"] - first_open) / abs(gap_val)).clip(0, 1)
    else:
        df["gap_fill_ratio"] = 0.0

    if n > 0 and df["timestamp_ns"].iloc[0] > 0:
        df["minutes_since_open"] = (df["timestamp_ns"] - df["timestamp_ns"].iloc[0]) / 60e9
    else:
        df["minutes_since_open"] = np.arange(n) * 0.25  # 근사

    # 오프닝 레인지 (첫 20바)
    or_bars = min(20, n)
    or_high = df["high"].iloc[:or_bars].max()
    or_low = df["low"].iloc[:or_bars].min()
    or_range = or_high - or_low
    if or_range > 0:
        df["opening_range_pos"] = ((df["close"] - or_low) / or_range)
        df.loc[:or_bars - 1, "opening_range_pos"] = 0.5
    else:
        df["opening_range_pos"] = 0.5

    # 세션 방향성
    session_net = df["close"] - first_open
    total_abs_change = df["change"].abs().cumsum().replace(0, np.nan)
    df["session_direction"] = (session_net / total_abs_change).fillna(0).clip(-1, 1)
    df["session_momentum"] = session_net

    return df[FEATURE_NAMES]
