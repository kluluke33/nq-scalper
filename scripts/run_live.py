"""
실전/페이퍼 트레이딩 시작 진입점.

세션 흐름:
  09:25  시스템 시작 → 브로커 연결, 경제지표 로드
  09:30  틱 수집 시작, 전략 웜업
  09:35  거래 시작
  10:30  거래 종료, 포지션 청산, 시스템 종료

사용법:
    python scripts/run_live.py                    # 기본 설정
    python scripts/run_live.py --paper             # 페이퍼 트레이딩
    python scripts/run_live.py --config path.yaml  # 커스텀 설정
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cold.calendar import load_events_from_file
from src.hot.broker_ibkr import IBKRAdapter
from src.hot.engine import TradingEngine
from src.strategy.ai_strategy import AIStrategy
from src.strategy.breakout_strategy import BreakoutStrategy
from src.strategy.manager import StrategyManager
from src.watchdog import Watchdog


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    log_file = log_cfg.get("file", "logs/trading.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )


def build_strategies(config: dict, strategy_manager: StrategyManager) -> None:
    strategy_cfg = config.get("strategy", {})

    ai = AIStrategy(strategy_cfg)
    model_path = strategy_cfg.get("ai", {}).get("model_path", "")
    if model_path:
        ai.load_model(model_path)
    strategy_manager.add(ai)

    breakout = BreakoutStrategy(strategy_cfg)
    strategy_manager.add(breakout)


def load_economic_calendar(config: dict, engine: TradingEngine) -> None:
    """경제지표 캘린더를 로드하여 세션 매니저에 등록."""
    cal_cfg = config.get("economic_calendar", {})
    events_file = cal_cfg.get("file", "config/economic_events.yaml")

    events = load_events_from_file(events_file)
    engine.session.set_economic_events(events)

    if events:
        log = logging.getLogger("calendar")
        for ev in events:
            log.info("  Blackout: %s (%s)", ev.name, ev.release_time.strftime("%H:%M ET"))


def run_engine(config: dict, shared_heartbeat) -> None:
    """Trading Process 진입점."""
    import time

    setup_logging(config)
    log = logging.getLogger("engine")

    broker = IBKRAdapter(config.get("broker", {}))
    engine = TradingEngine(config, broker)
    build_strategies(config, engine.strategy_manager)
    load_economic_calendar(config, engine)

    log.info("=" * 60)
    log.info("NQ Scalper starting")
    log.info("Session: %s", engine.session.summary())
    log.info("=" * 60)

    async def _run():
        try:
            await engine.start()
            while engine._running:
                shared_heartbeat.value = time.time()
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            log.info("Interrupted by user")
        finally:
            await engine.stop()

    asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser(description="NQ Scalper")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--paper", action="store_true", help="Use paper trading port (7497)")
    parser.add_argument("--no-watchdog", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.paper:
        config["broker"]["port"] = 7497

    setup_logging(config)

    if args.no_watchdog:
        shared_hb = multiprocessing.Value("d", 0.0)
        run_engine(config, shared_hb)
    else:
        watchdog = Watchdog(run_engine, config)
        watchdog.start()


if __name__ == "__main__":
    main()
