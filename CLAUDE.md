# NQ Scalper — Claude 작업 가이드

## 프로젝트 요약

E-mini Nasdaq-100(NQ) 선물 자동매매 시스템. 미국장 개장 직후 55분(09:35~10:30 ET)만 거래하며, LightGBM 기반 AI로 진입 시점을 판단하여 10~20포인트 스캘핑을 실행한다.

상세 프로젝트 문서: `docs/PROJECT.md`

## 핵심 규칙

### 거래 조건 (절대 변경 금지)

- 거래 시간: 09:35~10:30 ET (개장 후 5분 ~ 1시간)
- 경제지표 발표 전후 5분 블랙아웃 — 신규 진입만 차단, 기존 포지션은 조건부 청산
- 블랙아웃 포지션 처리: 수익 중이면 TP 또는 진입가 본전 탈출, 손실 중이면 최초 SL에서 탈출
- 10:27 신규 진입 차단, 10:29:30 전체 청산

### 아키텍처 원칙

- **핫패스/콜드패스 분리**: 거래 로직(src/hot/)은 단일 asyncio 루프, 로깅/저장(src/cold/)은 별도 스레드
- **핫패스에서 절대 금지**: DB 조회, 파일 읽기, 동기 블로킹, GC 유발 객체 대량 생성
- **리스크 게이트**: 인메모리 동기 체크만, 모든 주문은 RiskGate.allow()를 통과해야 함

### 전략 시스템 원칙

- 모든 전략은 `BaseStrategy`를 상속하고 `on_bar()` 구현
- 전략 추가 시 `@register_strategy` 데코레이터 사용
- 실시간과 학습의 피처 정의는 반드시 동일해야 함 (IncrementalFeatures.feature_names() == FEATURE_NAMES)
- 피처 수 변경 시: ai_strategy.py의 IncrementalFeatures, model/features.py의 build_features_from_bars, 학습된 모델 모두 동기화 필요

### AI 모델 원칙

- 현재 모델: LightGBM (추론 ~0.1ms)
- 피처: 30개 (ai_strategy.py의 IncrementalFeatures 참조)
- 라벨링: 양방향 (labeler.py의 label_bars_bidirectional)
- 학습 데이터 분할: 반드시 시간순 (walk-forward). 랜덤 분할 금지 — look-ahead bias 발생
- 모델 파일: src/model/models/ 에 타임스탬프 포함 저장, latest 심볼릭 링크 유지

## 디렉토리 구조

```
src/hot/          핫패스 — 지연 민감 (engine, tick_chart, executor, risk_gate, session, broker)
src/cold/         콜드패스 — 지연 허용 (tick_recorder, calendar, logger, monitor)
src/strategy/     전략 (base, ai_strategy, breakout, manager, market_regime)
src/model/        AI 모델 (features, labeler, train)
config/           설정 파일
scripts/          실행 스크립트 (run_live, train_model, retrain, download_databento, build_bars)
backtest/         백테스트
data/raw/         틱 CSV (Databento + IBKR 축적)
data/bars/        틱 차트 바 CSV
docs/             프로젝트 문서
```

## 설정

모든 설정은 `config/settings.yaml`에서 관리. 코드에 매직 넘버 하드코딩 금지.
경제지표 일정은 `config/economic_events.yaml`에 등록.

## 데이터

- 초기 학습 데이터: Databento NQ 1년 틱 (~$70)
- 매일 축적: 엔진 실행 시 TickRecorder가 data/raw/ticks_YYYYMMDD.csv에 자동 저장
- 형식: timestamp, price, size, side (Databento와 IBKR 동일 형식)
- 재학습: `python scripts/retrain.py` — Databento + IBKR 축적분 합쳐서 학습

## 실행 명령어

```bash
# 데이터 다운로드 (초기 1회)
python scripts/download_databento.py --months 12

# 틱 → 바 변환
python scripts/build_bars.py

# 모델 학습
python scripts/train_model.py --tune

# 백테스트
python -m backtest.runner --model src/model/models/lgbm_nq_latest.pkl

# 페이퍼 트레이딩
python scripts/run_live.py --paper

# 실전
python scripts/run_live.py

# 재학습 (축적 데이터 포함)
python scripts/retrain.py
```

## 코드 수정 시 주의사항

- 피처 추가/변경 시: `ai_strategy.py`의 IncrementalFeatures와 `model/features.py`의 build_features_from_bars 양쪽 모두 동기화
- Position 객체에는 entry_price, take_profit_price, stop_loss_price가 있음 — 진입 시 절대가격으로 계산되어 저장됨
- NQ 1포인트 = $5.00, 1틱 = 0.25포인트 = $1.25
- 브로커: IBKR paper port 7497, live port 7496
