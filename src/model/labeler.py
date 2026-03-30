"""
라벨링 모듈.

각 바 시점에서 미래 가격 변화를 측정하여 라벨을 생성.

라벨링 방식:
  "이 바 이후 N바 이내에 TP(+15pt)에 먼저 도달하는가, SL(-10pt)에 먼저 도달하는가?"

  → TP 먼저 도달: 1 (LONG 정답)
  → SL 먼저 도달: 0 (SHORT 정답)
  → 둘 다 미도달:  제외 (NaN) — 불확실한 샘플은 학습에서 제외
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def label_bars(
    bars_df: pd.DataFrame,
    tp_points: float = 15.0,
    sl_points: float = 10.0,
    max_bars_ahead: int = 50,
) -> pd.Series:
    """
    각 바에 대해 LONG 관점의 라벨을 생성.

    Args:
        bars_df: high, low, close 컬럼이 있는 DataFrame
        tp_points: 수익실현 포인트
        sl_points: 손절 포인트
        max_bars_ahead: 최대 탐색 바 수 (이내에 TP/SL 미도달 시 NaN)

    Returns:
        Series: 1.0 (LONG), 0.0 (SHORT), NaN (불확실)
    """
    n = len(bars_df)
    labels = np.full(n, np.nan, dtype=np.float64)

    closes = bars_df["close"].values
    highs = bars_df["high"].values
    lows = bars_df["low"].values

    for i in range(n - 1):
        entry = closes[i]
        tp_price = entry + tp_points
        sl_price = entry - sl_points

        # 미래 바들을 순서대로 탐색
        end = min(i + 1 + max_bars_ahead, n)
        for j in range(i + 1, end):
            hit_tp = highs[j] >= tp_price
            hit_sl = lows[j] <= sl_price

            if hit_tp and hit_sl:
                # 같은 바에서 둘 다 → 바 내 순서 알 수 없으므로 제외
                break
            elif hit_tp:
                labels[i] = 1.0  # LONG 정답
                break
            elif hit_sl:
                labels[i] = 0.0  # SHORT 정답
                break
            # 둘 다 아니면 다음 바로

    return pd.Series(labels, index=bars_df.index, name="label")


def label_bars_bidirectional(
    bars_df: pd.DataFrame,
    tp_points: float = 15.0,
    sl_points: float = 10.0,
    max_bars_ahead: int = 50,
) -> pd.Series:
    """
    양방향 라벨링.
    LONG과 SHORT 양쪽 모두 평가하여 더 빨리 도달하는 방향을 라벨로.

    Returns:
        Series: 1.0 (LONG 유리), 0.0 (SHORT 유리), NaN (불확실)
    """
    n = len(bars_df)
    labels = np.full(n, np.nan, dtype=np.float64)

    closes = bars_df["close"].values
    highs = bars_df["high"].values
    lows = bars_df["low"].values

    for i in range(n - 1):
        entry = closes[i]

        # LONG: TP = entry + tp, SL = entry - sl
        long_tp = entry + tp_points
        long_sl = entry - sl_points

        # SHORT: TP = entry - tp, SL = entry + sl
        short_tp = entry - tp_points
        short_sl = entry + sl_points

        long_hit_bar = -1
        short_hit_bar = -1

        end = min(i + 1 + max_bars_ahead, n)
        for j in range(i + 1, end):
            # LONG 체크
            if long_hit_bar < 0:
                if highs[j] >= long_tp:
                    long_hit_bar = j - i
                elif lows[j] <= long_sl:
                    long_hit_bar = -(j - i)  # 음수 = SL 도달 (실패)

            # SHORT 체크
            if short_hit_bar < 0:
                if lows[j] <= short_tp:
                    short_hit_bar = j - i
                elif highs[j] >= short_sl:
                    short_hit_bar = -(j - i)

            if long_hit_bar != 0 and short_hit_bar != 0:
                break

        # 판정
        long_success = long_hit_bar > 0
        short_success = short_hit_bar > 0

        if long_success and short_success:
            # 둘 다 성공 → 더 빨리 도달한 쪽
            labels[i] = 1.0 if long_hit_bar <= short_hit_bar else 0.0
        elif long_success:
            labels[i] = 1.0
        elif short_success:
            labels[i] = 0.0
        # 둘 다 실패 → NaN (제외)

    return pd.Series(labels, index=bars_df.index, name="label")
