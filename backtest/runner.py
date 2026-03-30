"""
백테스트 실행기.

과거 바 데이터에 대해 전체 거래 파이프라인을 시뮬레이션.
수수료, 슬리피지를 포함한 실제 수익을 계산.

사용법:
    python -m backtest.runner --model src/model/models/lgbm_nq_latest.pkl
    python -m backtest.runner --model ... --slippage 0.5 --commission 1.70
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.features import FEATURE_NAMES, build_features_from_bars
from src.strategy.ai_strategy import IncrementalFeatures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


NQ_POINT_VALUE = 5.0  # NQ 미니 1포인트 = $5


@dataclass
class BacktestConfig:
    model_path: str = ""
    tp_points: float = 15.0
    sl_points: float = 10.0
    trailing_trigger: float = 8.0
    trailing_offset: float = 4.0
    min_confidence: float = 0.65
    slippage_points: float = 0.25   # 편도 슬리피지 (1틱)
    commission_usd: float = 1.70    # 왕복 수수료
    max_bars_hold: int = 100        # 최대 보유 바 수


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    direction: int           # 1=LONG, -1=SHORT
    entry_price: float
    exit_price: float
    pnl_points: float
    pnl_usd: float
    exit_reason: str
    confidence: float
    bars_held: int


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    total_pnl_usd: float = 0.0
    total_pnl_points: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    total_commission: float = 0.0
    max_drawdown_usd: float = 0.0

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        return self.win_count / self.trade_count if self.trade_count > 0 else 0

    @property
    def avg_win(self) -> float:
        wins = [t.pnl_usd for t in self.trades if t.pnl_usd > 0]
        return np.mean(wins) if wins else 0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl_usd for t in self.trades if t.pnl_usd <= 0]
        return np.mean(losses) if losses else 0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_usd for t in self.trades if t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in self.trades if t.pnl_usd < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def net_pnl(self) -> float:
        return self.total_pnl_usd - self.total_commission


def run_backtest(
    sessions: list[pd.DataFrame],
    prev_day_closes: list[float],
    config: BacktestConfig,
) -> BacktestResult:
    """세션별로 백테스트 실행."""
    import lightgbm as lgb

    model = lgb.Booster(model_file=config.model_path)
    result = BacktestResult()

    for session_idx, (bars_df, prev_close) in enumerate(zip(sessions, prev_day_closes)):
        if len(bars_df) < 30:
            continue

        features_df = build_features_from_bars(bars_df, prev_close)
        features_arr = features_df.values
        closes = bars_df["close"].values
        highs = bars_df["high"].values
        lows = bars_df["low"].values
        n = len(bars_df)

        i = 20  # 오프닝 레인지(20바) 이후부터 거래
        while i < n - 1:
            # 추론
            prob = model.predict(features_arr[i:i + 1])[0]

            # 신호 판단
            if prob > (0.5 + config.min_confidence / 2):
                direction = 1
                confidence = (prob - 0.5) * 2
            elif prob < (0.5 - config.min_confidence / 2):
                direction = -1
                confidence = (0.5 - prob) * 2
            else:
                i += 1
                continue

            # 진입 (슬리피지 적용)
            entry_price = closes[i] + config.slippage_points * direction

            # TP/SL 가격
            tp_price = entry_price + config.tp_points * direction
            sl_price = entry_price - config.sl_points * direction

            # 포지션 시뮬레이션
            highest_pnl = 0.0
            exit_bar = -1
            exit_price = 0.0
            exit_reason = ""

            for j in range(i + 1, min(i + 1 + config.max_bars_hold, n)):
                pnl_points = (closes[j] - entry_price) * direction

                # TP 체크
                if direction == 1 and highs[j] >= tp_price:
                    exit_bar = j
                    exit_price = tp_price
                    exit_reason = "TAKE_PROFIT"
                    break
                elif direction == -1 and lows[j] <= tp_price:
                    exit_bar = j
                    exit_price = tp_price
                    exit_reason = "TAKE_PROFIT"
                    break

                # SL 체크
                if direction == 1 and lows[j] <= sl_price:
                    exit_bar = j
                    exit_price = sl_price
                    exit_reason = "STOP_LOSS"
                    break
                elif direction == -1 and highs[j] >= sl_price:
                    exit_bar = j
                    exit_price = sl_price
                    exit_reason = "STOP_LOSS"
                    break

                # 트레일링 스탑
                highest_pnl = max(highest_pnl, pnl_points)
                if highest_pnl >= config.trailing_trigger:
                    trailing_level = highest_pnl - config.trailing_offset
                    if pnl_points <= trailing_level:
                        exit_bar = j
                        exit_price = closes[j] - config.slippage_points * direction
                        exit_reason = "TRAILING_STOP"
                        break

            # 최대 보유 초과 시 강제 청산
            if exit_bar < 0:
                exit_bar = min(i + config.max_bars_hold, n - 1)
                exit_price = closes[exit_bar] - config.slippage_points * direction
                exit_reason = "TIME_EXIT"

            # 수익 계산
            pnl_points = (exit_price - entry_price) * direction
            pnl_usd = pnl_points * NQ_POINT_VALUE

            trade = Trade(
                entry_bar=i,
                exit_bar=exit_bar,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_points=pnl_points,
                pnl_usd=pnl_usd,
                exit_reason=exit_reason,
                confidence=confidence,
                bars_held=exit_bar - i,
            )
            result.trades.append(trade)
            result.total_pnl_usd += pnl_usd
            result.total_pnl_points += pnl_points
            result.total_commission += config.commission_usd

            if pnl_usd > 0:
                result.win_count += 1
            else:
                result.loss_count += 1

            # 다음 진입은 청산 이후부터
            i = exit_bar + 1

    # 최대 낙폭 계산
    if result.trades:
        equity = np.cumsum([t.pnl_usd - config.commission_usd for t in result.trades])
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        result.max_drawdown_usd = float(np.max(drawdown))

    return result


def print_report(result: BacktestResult, config: BacktestConfig) -> None:
    """백테스트 결과 리포트."""
    log.info("=" * 60)
    log.info("BACKTEST REPORT")
    log.info("=" * 60)
    log.info("Config: TP=%.1f SL=%.1f slip=%.2f comm=$%.2f",
             config.tp_points, config.sl_points,
             config.slippage_points, config.commission_usd)
    log.info("-" * 60)
    log.info("Total Trades:     %d", result.trade_count)
    log.info("Win Rate:         %.1f%%", result.win_rate * 100)
    log.info("Wins / Losses:    %d / %d", result.win_count, result.loss_count)
    log.info("-" * 60)
    log.info("Gross PnL:        $%.2f (%.1f pts)", result.total_pnl_usd, result.total_pnl_points)
    log.info("Total Commission: $%.2f", result.total_commission)
    log.info("Net PnL:          $%.2f", result.net_pnl)
    log.info("-" * 60)
    log.info("Avg Win:          $%.2f", result.avg_win)
    log.info("Avg Loss:         $%.2f", result.avg_loss)
    log.info("Profit Factor:    %.2f", result.profit_factor)
    log.info("Max Drawdown:     $%.2f", result.max_drawdown_usd)
    log.info("-" * 60)

    # 청산 사유별 통계
    reasons = {}
    for t in result.trades:
        if t.exit_reason not in reasons:
            reasons[t.exit_reason] = {"count": 0, "pnl": 0.0}
        reasons[t.exit_reason]["count"] += 1
        reasons[t.exit_reason]["pnl"] += t.pnl_usd

    log.info("Exit Reasons:")
    for reason, stats in sorted(reasons.items()):
        log.info("  %-15s %3d trades  $%.2f", reason, stats["count"], stats["pnl"])

    # 일별 PnL
    if result.trades:
        log.info("-" * 60)
        log.info("Per-Trade PnL Distribution:")
        pnls = [t.pnl_usd for t in result.trades]
        log.info("  Min:    $%.2f", min(pnls))
        log.info("  25th:   $%.2f", np.percentile(pnls, 25))
        log.info("  Median: $%.2f", np.median(pnls))
        log.info("  75th:   $%.2f", np.percentile(pnls, 75))
        log.info("  Max:    $%.2f", max(pnls))

    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Backtest NQ scalping model")
    parser.add_argument("--model", required=True, help="Model file path")
    parser.add_argument("--bars-dir", default="data/bars")
    parser.add_argument("--tp", type=float, default=15.0)
    parser.add_argument("--sl", type=float, default=10.0)
    parser.add_argument("--slippage", type=float, default=0.25)
    parser.add_argument("--commission", type=float, default=1.70)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    args = parser.parse_args()

    config = BacktestConfig(
        model_path=args.model,
        tp_points=args.tp,
        sl_points=args.sl,
        slippage_points=args.slippage,
        commission_usd=args.commission,
        min_confidence=args.min_confidence,
    )

    # 바 데이터 로드
    from scripts.train_model import load_sessions
    sessions, prev_day_closes = load_sessions(args.bars_dir)

    # 백테스트 실행
    log.info("Running backtest with %d sessions...", len(sessions))
    result = run_backtest(sessions, prev_day_closes, config)

    # 리포트
    print_report(result, config)


if __name__ == "__main__":
    main()
