# NQ Scalper — E-mini Nasdaq-100 자동매매 시스템

## 프로젝트 개요

E-mini Nasdaq-100(NQ) 선물을 대상으로 틱 차트 기반 AI 자동매매를 실행하는 시스템.
미국장 개장 직후 55분 동안만 거래하며, 10~20포인트(40~80틱) 구간의 스캘핑을 목표로 한다.

---

## 거래 조건

| 항목 | 내용 |
|------|------|
| 종목 | E-mini Nasdaq-100 (NQ) |
| 거래 시간 | 09:35~10:30 ET (개장 후 5분 ~ 1시간) |
| 목표 수익 | 10~20포인트 (기본 15pt, $75/계약) |
| 손절 | 10포인트 ($50/계약) |
| 1포인트 가치 | $5.00 (미니 1계약) |
| 1틱 | 0.25포인트 = $1.25 |

### 거래 금지 시간

- 개장 후 5분 이내 (09:30~09:35 ET) — 웜업 구간
- 경제지표 발표 전후 5분씩 (총 10분) — 블랙아웃
- 거래 종료 3분 전 — 신규 진입 차단
- 거래 종료 30초 전 — 포지션 전체 청산

### 블랙아웃 포지션 처리

즉시 청산하지 않고 조건부 관리:

- **수익 중**: TP 도달 시 청산, 진입가 되돌림 시 본전 탈출
- **손실 중**: 최초 SL 도달 시 시장가 탈출

---

## 플랫폼 & 기술 스택

| 항목 | 선택 | 이유 |
|------|------|------|
| 브로커 | Interactive Brokers | 왕복 $1.70, API 성숙 |
| 데이터 (초기) | Databento 1년 틱 | ~$70, Python SDK, side 포함 |
| 데이터 (축적) | IBKR 실시간 자동 저장 | 무료, 매일 자동 |
| 언어 | Python | AI/ML 생태계 |
| AI 모델 | LightGBM | 추론 ~0.1ms, 소규모 데이터 최적 |
| 브로커 API | ib_insync | 비동기, 성숙 |

---

## 시스템 아키텍처

### 핫패스 / 콜드패스 분리

```
Trading Process (핫패스, 단일 asyncio 루프)
  틱 수신 → 세션 체크 → 틱 차트 → 예비 추론 → 전략 → 리스크 → 주문
  목표 지연: < 1ms

Support Process (콜드패스, 별도 스레드/프로세스)
  로깅, 모니터링, 알림, 틱 저장

Watchdog (독립 프로세스)
  Trading Process 생존 감시, 비정상 시 청산 + 재시작
```

### 세션 흐름

```
09:25  시스템 시작 → 브로커 연결
09:30  WARMUP — 틱 수집 + 전략 웜업
09:35  ACTIVE — 거래 시작
       (지표 있으면 BLACKOUT 진입/복귀)
10:27  신규 진입 차단
10:29:30 포지션 전체 청산
10:30  DONE — 시스템 종료, 일일 리포트
```

---

## 디렉토리 구조

```
nq-scalper/
├── config/
│   ├── settings.yaml              전체 설정
│   └── economic_events.yaml       경제지표 발표 일정
├── src/
│   ├── hot/                       핫패스 (지연 민감)
│   │   ├── engine.py              메인 이벤트 루프
│   │   ├── tick_chart.py          적응형 틱 차트 바 생성
│   │   ├── executor.py            주문 실행 + 청산 관리
│   │   ├── risk_gate.py           인메모리 리스크 체크
│   │   ├── session.py             세션/블랙아웃 관리
│   │   └── broker_ibkr.py         IBKR 어댑터
│   ├── cold/                      콜드패스 (지연 허용)
│   │   ├── tick_recorder.py       실시간 틱 자동 저장
│   │   ├── calendar.py            경제지표 캘린더
│   │   ├── logger.py              거래 기록
│   │   └── monitor.py             상태 감시
│   ├── model/                     AI 모델
│   │   ├── features.py            피처 정의 + 일괄 생성
│   │   ├── labeler.py             라벨링 (양방향)
│   │   ├── train.py               학습 파이프라인
│   │   └── models/                저장된 모델 파일
│   ├── strategy/                  전략
│   │   ├── base.py                전략 인터페이스 (BaseStrategy)
│   │   ├── ai_strategy.py         AI 전략 (LightGBM + 사전 추론)
│   │   ├── breakout_strategy.py   브레이크아웃 전략
│   │   ├── manager.py             전략 등록/선택/합성
│   │   ├── market_regime.py       시장 상태 판별
│   │   └── custom_template.py     커스텀 전략 템플릿
│   └── watchdog.py                프로세스 감시
├── backtest/
│   └── runner.py                  백테스트 실행기
├── scripts/
│   ├── download_databento.py      Databento 틱 다운로드
│   ├── download_history.py        IBKR 과거 데이터 다운로드
│   ├── build_bars.py              틱 → 바 변환
│   ├── train_model.py             모델 학습
│   ├── retrain.py                 모델 재학습 (축적 데이터 포함)
│   └── run_live.py                실전/페이퍼 트레이딩 실행
├── data/
│   ├── raw/                       틱 CSV (Databento + IBKR)
│   └── bars/                      틱 차트 바 CSV
└── tests/
```

---

## AI 모델

### 모델: LightGBM

선택 근거:
- 추론 속도 ~0.1ms (딥러닝 대비 10~50배 빠름)
- 2.7만~5.5만 샘플로 학습 가능 (LSTM/Transformer는 10만+ 필요)
- 피처 중요도 확인 가능 (해석 가능)
- 과적합 제어 쉬움

### 피처 30개

| 그룹 | 피처 | 개수 |
|------|------|------|
| 가격 구조 | close, change, EMA(8/21), MACD, RSI, range, body, close_position | 9 |
| 주문 흐름 | delta, buy_ratio, cum_delta, duration_ms | 4 |
| 일중 컨텍스트 | day_position, day_high/low, dist_ema_fast/slow | 5 |
| 시퀀스 대체 | momentum_3/5bar, delta_sum_3/5bar, bullish_ratio, range_ratio | 6 |
| 개장 직후 특화 | gap, gap_fill, minutes_since_open, opening_range_pos, session_direction/momentum | 6 |

### 라벨링

양방향 평가 (bidirectional):
- 각 바 시점에서 LONG/SHORT 양쪽의 TP/SL 도달 속도를 비교
- LONG이 빨리 TP 도달 → label=1, SHORT이 빨리 → label=0
- 둘 다 실패 → 제외 (NaN)

### 사전 추론 (Pre-inference)

```
틱 #480 (96% 채움) → 예비 피처 스냅샷 → 예비 추론 → 결과 캐싱
틱 #500 (바 완성)  → 최종 피처 계산 → 캐시와 비교
                     변화 미미 → 캐시 결과 즉시 사용 (추론 스킵)
                     변화 큼   → 최종 추론 1회
```

### 모델 진화 로드맵

| Phase | 기간 | 모델 | 데이터 |
|-------|------|------|--------|
| 1 | 0~6개월 | LightGBM | Databento 1년 + IBKR 축적 |
| 2 | 6~12개월 | LightGBM + 1D-CNN 앙상블 | 5만+ 샘플 |
| 3 | 12개월+ | LSTM / TFT 검토 | 10만+ 샘플 |

---

## 전략 시스템

### 전략 모드 (settings.yaml → strategy.active)

| 모드 | 동작 |
|------|------|
| `ai` | AI 전략만 사용 (고정) |
| `breakout` | 브레이크아웃 전략만 사용 (고정) |
| `composite` | 복수 전략 합성 (만장일치/가중평균 등) |
| `auto` | 시장 상태(regime)에 따라 자동 전환 |

### auto 모드 — 시장 상태별 전략 매핑

| 시장 상태 | 판별 기준 | 적용 전략 | 보정 |
|-----------|----------|-----------|------|
| TRENDING | 효율 비율 > 0.4 | AI | TP ×1.3 |
| RANGING | 효율 비율 ≤ 0.4 | Breakout | 없음 |
| VOLATILE | ATR 2배 초과 | AI | 확신도 ×0.8, SL ×1.5 |
| QUIET | 바 60초+ | 진입 보류 | — |

### 커스텀 전략 추가

1. `custom_template.py` 복사
2. `BaseStrategy` 상속, `name` 변경, `on_bar()` 구현
3. `@register_strategy` 데코레이터 추가
4. `settings.yaml`에서 active로 설정 또는 composite에 추가

---

## 틱 차트

### 적응형 틱 수 (Adaptive Tick Size)

시장 활동량(TPS)에 따라 바 크기를 자동 조절:

```
틱 수 = smoothed_TPS × target_bar_duration_sec

개장 직후 설정:
  target: 15초, min: 300틱, max: 1500틱
  TPS 50 → 750틱/바, TPS 100 → 1500틱/바
```

고정 모드: `adaptive: false` → 항상 500틱

---

## 청산 로직

### 통상 거래 (ACTIVE)

1. **손절**: PnL ≤ -SL → 시장가 청산
2. **수익실현**: PnL ≥ TP → 시장가 청산
3. **트레일링 스탑**: PnL ≥ 8pt 시 시작, 최고점-4pt로 추적

### 블랙아웃 (경제지표 전후)

- 수익 중: TP 도달 → 청산 / 진입가 되돌림 → 본전 탈출
- 손실 중: 최초 SL 도달 → 시장가 탈출
- 신규 진입만 차단, 기존 포지션은 조건부 관리

### 세션 종료

- 10:27: 신규 진입 차단
- 10:29:30: 전체 포지션 시장가 청산

---

## 리스크 관리

| 항목 | 설정 | 이유 |
|------|------|------|
| 최대 계약 수 | 2 | 소규모 시작 |
| 일일 손실 한도 | $200 | 일일 최대 허용 손실 |
| 연속 손실 한도 | 3회 | 3연패 시 쿨다운 |
| 쿨다운 | 10분 | 55분 윈도우에 맞춤 |

---

## 데이터 파이프라인

### 초기 데이터 (Databento)

```bash
export DATABENTO_API_KEY="your-key"
python scripts/download_databento.py --months 12
python scripts/build_bars.py --tick-count 500
```

### 매일 자동 축적 (IBKR)

엔진 실행 시 `TickRecorder`가 매 틱을 `data/raw/ticks_YYYYMMDD.csv`에 자동 저장.
Databento 데이터와 동일한 형식 (timestamp, price, size, side).

### 모델 재학습

```bash
# 축적된 데이터를 포함하여 재학습
python scripts/retrain.py

# 하이퍼파라미터 튜닝 포함
python scripts/retrain.py --tune
```

---

## 실행 방법

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비 + 모델 학습

```bash
python scripts/download_databento.py --months 12
python scripts/build_bars.py
python scripts/train_model.py --tune
```

### 3. 백테스트

```bash
python -m backtest.runner --model src/model/models/lgbm_nq_latest.pkl
```

### 4. 페이퍼 트레이딩

```bash
# IBKR TWS/Gateway 실행 필요 (포트 7497)
python scripts/run_live.py --paper
```

### 5. 실전 트레이딩

```bash
# IBKR TWS/Gateway 실행 필요 (포트 7496)
python scripts/run_live.py
```

---

## 진행 현황

### 완료

- [x] 시스템 아키텍처 설계 (핫패스/콜드패스 분리)
- [x] 세션 관리 (09:35~10:30 ET, 블랙아웃)
- [x] 적응형 틱 차트 바 생성
- [x] 전략 플러그인 시스템 (등록/합성/자동 전환)
- [x] AI 전략 (LightGBM, 사전 추론, 30개 피처)
- [x] 브레이크아웃 전략
- [x] 시장 상태 판별 (RegimeDetector)
- [x] 주문 실행 + 청산 로직 (TP/SL/트레일링/블랙아웃)
- [x] 리스크 게이트
- [x] IBKR 어댑터
- [x] Watchdog
- [x] 경제지표 캘린더
- [x] 피처 엔지니어링 (30개)
- [x] 라벨링 모듈 (양방향)
- [x] 모델 학습 파이프라인
- [x] 백테스트 프레임워크
- [x] Databento 데이터 다운로더
- [x] 실시간 틱 자동 저장 (TickRecorder)
- [x] 재학습 스크립트

### 미완료

- [ ] Databento 계정 생성 + API 키 발급
- [ ] 1년 틱 데이터 다운로드
- [ ] 모델 학습 + 백테스트 검증
- [ ] IBKR 페이퍼 트레이딩 테스트
- [ ] 단위 테스트 작성
- [ ] Telegram/Slack 알림 연동
- [ ] Grafana 모니터링 대시보드
- [ ] 성과 분석 리포트 자동화
- [ ] 모델 진화 Phase 2 (1D-CNN 앙상블)
