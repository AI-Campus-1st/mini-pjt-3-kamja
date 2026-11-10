# 생성형 AI 활용 광고 소비자 반응 예상 RAG 챗봇

> AI 이미지를 쓰려는 신입 마케터·기획자에게, **비슷한 국내 AI 광고 사례의 실제 유튜브 댓글 반응**을 근거로 "어떤 반응이 나왔는지"를 보여 주는 챗봇. 예상 등급은 LLM이 아니라 **코드 규칙**으로 정한다.

- 과제: 미니프로젝트 3 — 공공 문서·웹검색 결합 RAG 기반 질의응답 시스템 (가이드 주제 C)
- 기간: 2026-11-03 ~ 2026-11-10
- **리포트: [`report/final_report.md`](report/final_report.md)**

| 구분 | 결과 |
|---|---|
| 지식베이스 | 유튜브 댓글 정제 2,290개(사례 11개, 자격 7개) + 제공 뉴스 16건 + 정부 안내서 2종 → 청크 564개 |
| 검색 (32문항) | Hit@1 · Hit@3 · MRR = 0.719 · 0.781 · 0.770 → **0.812 · 0.875 · 0.846** |
| 답변 (32문항) | 정확 43.8% → **50.0%**, 오답 7 → 4 |
| 답변 불가 (8문항) | 8/8 거절, 잘못 거절 0 |

## 1. 폴더 구조

```
.
├── collector/   ① 수집: 유튜브 댓글 · 제공 뉴스 · 정부 PDF → store/docs.db
├── indexer/     ② 인덱싱: 파싱 · 정제 · 댓글 분류(GPT) · 사례 요약 · 청킹 · 임베딩 → store/
├── store/       ③ 저장소: chroma/ (벡터 DB) · docs.db (원문·분류 결과·청크)
├── rag/         검색 · 범위 판정 · 특징 추출 · 등급 규칙 · chain.py (챗봇과 평가가 함께 씀)
├── app/         ④ Streamlit 챗봇 (조회만)
├── eval/        평가 하네스 · 평가셋 · 결과        → eval/README.md
├── report/      final_report.md + evidence/(근거 자료)
├── notebooks/   EDA · 뉴스 선별 · API 시험
└── data/        원본 (제공 뉴스 묶음, 정부 PDF) — 저장소에 포함하지 않음
```

계층은 한 방향이다: **collector → indexer → store → app**. 챗봇(app)은 문서 임베딩을 만들거나 인덱스에 쓰지 않고, 질문 임베딩만 만들어 조회한다. 챗봇과 평가 하네스는 같은 `rag/chain.py`를 부른다.

**collector의 CSV 4개는 사람이 정한 "수집 대상 목록"이다.** 수집 결과는 `store/docs.db`에 저장된다.

| 파일 | 내용 |
|---|---|
| `collector/cases.csv` | 사례 카드 11개 (특징 4개, 광고 내용 한 줄, 사실 근거 URL) |
| `collector/case_videos.csv` | 사례별 댓글 수집 영상 (`use=Y`만 수집) |
| `collector/gov_docs.csv` | 정부 안내서 2종의 제목·발행처·발행일·URL·이용 조건 |
| `collector/news_selected.csv` | 제공 뉴스에서 고른 16건 (선정 이유는 `report/evidence/news_shortlist.csv`) |

## 2. 준비

모든 명령은 **프로젝트 루트**에서 실행한다 (작성 환경: macOS, Python 3).

```bash
python -m pip install -r requirements.txt
cp .env.example .env        # OPENAI_API_KEY, YOUTUBE_API_KEY 를 채운다 (키는 .env에만)
```

- 임베딩 모델 KURE-v1(`nlpai-lab/KURE-v1`, 약 2.3GB)은 처음 실행할 때 내려받는다. 한 번 받은 뒤에는 `HF_HUB_OFFLINE=1`을 붙이면 Hub 확인 없이 실행된다.
- `data/`와 `store/`는 저장소에 없다. 아래 순서로 다시 만든다. 원본은 직접 준비한다.
  - `data/provided/`: 과제 제공 뉴스 묶음 CSV (`02_AI반도체_묶음_…csv`, UTF-8 BOM)
  - `data/gov/`: 정부 안내서 PDF 2종. 내려받을 곳은 `collector/gov_docs.csv`의 `url`, 저장 파일 이름은 `file_name`

## 3. 실행 순서 (계층별)

### ① collector — 수집

```bash
python -m collector.collect_youtube     # 유튜브 댓글 (작성자 이름·채널 ID·프로필 저장 안 함, @닉네임 → @사용자)
python -m collector.load_provided       # 선별 뉴스 16건 본문을 묶음 CSV에서 원문 그대로 적재
python -m collector.load_gov_docs       # 정부 안내서 목록·PDF 경로 적재
```

`collect_youtube --only case_01`로 한 사례만 먼저 시험할 수 있다.

### ② indexer — 정제·분류·청킹·임베딩

```bash
python -m indexer.parse                                     # 정부 PDF → 쪽별 글자 (머리글·쪽 번호 줄 제거)
python -m indexer.clean                                     # 댓글 정제 → report/evidence/clean_stats.csv
python -m indexer.label --tag exp1b_casedesc --casedesc     # GPT 댓글 분류 (gpt-4.1-mini, OpenAI 비용 발생)
python -m indexer.case_summary --tag exp1b_casedesc         # 사례별 비율·자격·주된 이유 (코드 계산, LLM 없음)
python -m indexer.chunk --tag exp1b_casedesc                # 4종 문서 청킹 → docs.db chunks
python -m indexer.embed                                     # KURE-v1 임베딩 → store/chroma (컬렉션 kb_kure_v1)
python -m indexer.add_display_names                         # 화면 사례명 메타 추가 (임베딩·본문 그대로)
python -m indexer.add_year_meta                             # 기간 필터용 발행 연도 메타 추가
python -m indexer.check_store                               # 적재 검증: 청크 수 대조, 메타 결손, 샘플 검색
```

- 분류 기준은 [`indexer/codebook.md`](indexer/codebook.md)(v2). `--casedesc`는 사례별 광고 내용 한 줄을 함께 넣는 설정으로, 최종 분류에 쓴 설정이다.
- 같은 태그로 이미 분류한 댓글은 건너뛰므로, 중간에 끊겨도 같은 명령으로 이어서 실행한다.
- `embed`는 다시 실행하면 컬렉션을 지우고 새로 만든다.

### ③ store — 저장소 (실행 없음)

| 위치 | 내용 |
|---|---|
| `store/docs.db` | 원문(`youtube_comments`, `news_provided`, `gov_docs`, `gov_pages`), 정제·분류(`comment_filter`, `comment_label`), `case_summary`, `chunks` |
| `store/chroma/` | 청크 564개 + 메타데이터 (출처·발행일·수집 일시·문서 제목·사례 특징 등). 검색은 공통 청크 + 댓글 시간순 안만 사용 |

### ④ app — 챗봇

```bash
HF_HUB_OFFLINE=1 streamlit run app/main.py
```

| 탭 | 내용 |
|---|---|
| 챗봇 | 멀티턴, 문장 단위 스트리밍, 기간·문서 종류·업종 필터, 번호 출처 목록 + 원문 보기, 예상 답변의 근거 사례 표 |
| 기획서 확인 (심화) | 기획서를 붙여 넣거나 올리면(PDF·Word·HWPX) AI 사용 항목별로 비슷한 사례와 등급을 보여 줌. 저장·임베딩하지 않음 |
| 성능 평가 | 실험별 Hit@k·MRR, 유형별 정확도(개선 전후), 1위 유사도 분포 등 (`eval/results`·채점표를 읽기만 함) |

## 4. 평가

자세한 내용은 [`eval/README.md`](eval/README.md), 결과 목차는 [`eval/results/README.md`](eval/results/README.md).

```bash
HF_HUB_OFFLINE=1 python -m eval.run_eval --tag <새 태그>        # 검색 Hit@1·Hit@3·MRR + 의도·특징·등급
HF_HUB_OFFLINE=1 python -m eval.check_answers --tag <새 태그>   # 40문항 답변 → report/evidence/<태그>.md (사람 채점)
HF_HUB_OFFLINE=1 python -m eval.holdout --tag <새 태그>         # 숨겨 두기 검증
```

- 평가셋: `eval/eval_set.jsonl` 40문항 (예상 10 · 사실 10 · 비교 6 · 종합 6 · 답변 불가 8)
- 같은 태그로 돌리면 기존 결과를 덮어쓴다. 다시 잴 때는 새 태그를 쓴다.
- LLM 단계(특징 추출·범위 판정·답변 문장)는 temperature 0에서도 실행마다 다를 수 있다.


결과는 `report/evidence/check_layers.md`, 원자료는 `eval/results/check_layers_detail.csv`.

## 6. 자료 이용 조건과 주의

- **YouTube 댓글:** YouTube API 정책상 로그인 없이 받은 데이터는 30일 안에 삭제하거나 다시 받아야 한다([Developer Policies III.E.4](https://developers.google.com/youtube/terms/developer-policies)). 이 프로젝트의 댓글은 2026-11-04에 수집했다. 다시 수집하면 댓글이 바뀌어 리포트 수치와 달라질 수 있다.

- **공개 저장소의 댓글 원문:** 검수 CSV·답변 기록·답변 원문 md에 들어 있던 유튜브 댓글 원문은 `(댓글 원문 비공개)`로 가렸다. `comment_id`, 사람·GPT 라벨, 문항·등급·사례·요약 구절은 그대로라 수치 근거는 남는다. 원문은 위 순서로 다시 수집하면 `comment_id`로 대조할 수 있다.

- **개인정보:** 댓글 작성자 이름·채널 ID·프로필은 저장하지 않는다. 기획서 확인 탭에 올린 문서는 저장하지 않지만, 항목 추출을 위해 OpenAI API로 전송된다.

- **정부 안내서:** 공공누리 제4유형(출처표시·상업적 이용금지·변경금지). 비상업 과제로 출처를 표시하고 검색 근거로만 쓴다.

- **예상 결과의 의미:** "관찰됨"은 비슷한 사례의 유튜브 댓글에서 관찰된 반응이지, 소비자 전체의 반응이나 맞힘 보장이 아니다. 최종 판단은 사람이 한다.

- **API 키:** `.env`에만 두고 코드·리포트·스크린샷에 넣지 않는다 (`.env`는 `.gitignore`에 등록).

## 7. 문서 안내

| 문서 | 내용 |
|---|---|
| [`report/final_report.md`](report/final_report.md) | 최종 리포트 (가이드 6항 1~10항) |
| [`report/evidence/`](report/evidence/) | 리포트 근거: 심화 설계(결과 전 고정), 실패 메모, 채점한 답변 원문, 구조 검사, 정제·사례·뉴스 CSV |
| [`indexer/codebook.md`](indexer/codebook.md) | 댓글 분류 기준표 v2 |
| [`notebooks/eda.ipynb`](notebooks/eda.ipynb) | 문서 탐색 결과 모음 (Day 6 재정리) |
| [`eval/README.md`](eval/README.md) | 평가 하네스 안내 |
