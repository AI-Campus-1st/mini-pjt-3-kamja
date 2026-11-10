# eval — 평가 하네스

- 모든 명령은 **프로젝트 루트**에서 실행한다. KURE-v1을 내려받았으면 `HF_HUB_OFFLINE=1`을 붙여 Hub 확인 없이 실행한다.
- 챗봇과 같은 `rag/chain.py`를 부른다(구조 검사 B항). 검색 지표만 재는 스크립트는 `rag.retriever.search`도 쓴다.
- 결과는 `eval/results/<태그>.json`(요약)과 `<태그>_detail.csv`(문항별)로 저장한다. 지난 결과는 단계별 하위 폴더(`1_label/`~`6_day4_checks/`)에 묶여 있고, 새 결과는 `results/` 바로 아래에 생긴다. 태그별 설명은 [`results/README.md`](results/README.md).
- store는 읽기만 한다. 인덱스를 다시 만들지 않는다.

## 0. 폴더 구성

```
eval/
├── eval_set.jsonl   평가셋 40문항 (가이드 6항이 지정한 경로라 그대로 둠)
├── *.py             평가 스크립트 18개 (python -m eval.<이름>으로 실행)
├── probes/          평가셋 밖 확인 질문 (평가 오염 확인)
├── plans/           심화 더미 기획서 + 정답표
├── grading/         사람이 직접 한 검수·채점
└── results/         실험·측정 결과 (단계별 하위 폴더, results/README.md)
```

스크립트는 하위 폴더로 나누지 않았다. 나누면 스크립트마다 프로젝트 루트 계산(`Path(__file__).parents[1]`)과 실행 명령이 바뀌기 때문이다.

## 1. 평가셋·정답 데이터 (직접 작성)

| 파일 | 내용 | 쓰는 스크립트 |
|---|---|---|
| `eval_set.jsonl` | **평가셋 40문항** (예상 10 · 사실 10 · 비교 6 · 종합 6 · 답변 불가 8) | 대부분 |
| `probes/extra_compare.jsonl` | 평가셋 밖 비교 질문 V01~V04 (평가 오염 확인) | `check_answers --set` |
| `probes/extra_fact.jsonl` | 평가셋 밖 사실 질문 Z01~Z03 (exp13 확인) | `check_answers --set` |
| `probes/feat_probe.jsonl` | 평가셋 밖 특징 확인 질문 Y01~Y06 (exp9 확인, 결과 보기 전 작성) | `feat_probe` |
| `probes/synth_probe.jsonl` | 평가셋 밖 종합 판정 확인 질문 W·N (exp10 확인) | `synth_check` |
| `plans/plan_A~F.md` | 심화 더미 기획서 6개 (A·B·C 첫 측정, D·E·F 개선 판정 전용) | `plan_eval` |
| `plans/plan_gold.jsonl`, `plans/plan_gold2.jsonl` | 더미 A·B·C / D·E·F 정답표 | `plan_eval`, `check_plan_gold` |

## 2. 사람 검수·채점 (`grading/`, 직접 작성)

| 파일 | 내용 |
|---|---|
| `grading/label_check.csv` | 사람 검수 100개 (정제 후 96개로 측정, 분류 개선용) |
| `grading/label_check2.csv` | 새 검수 40개 (최종 일치율, 과적합 확인용) |
| `grading/day5_grading.csv` | 개선 전 40문항 답변 채점 (대시보드가 읽음) |
| `grading/final_grading.csv` | 개선 후 40문항 답변 채점 (대시보드가 읽음) |
| `grading/exp10_grading.csv`, `grading/exp13_grading.csv` | 개선 실험별 겨냥 문항 채점 |

## 3. 스크립트

### ① 분류 (리포트 6-1)

| 스크립트 | 하는 일 | 실행 예 |
|---|---|---|
| `sample_label_check.py` | 검수용 100개 무작위 추출 (seed 고정) → `grading/label_check.csv` | `python -m eval.sample_label_check` |
| `sample_label_check2.py` | 새 검수 40개 추출 (seed 43, 기존 100개 제외) → `grading/label_check2.csv` | `python -m eval.sample_label_check2` |
| `label_agreement.py` | 사람 검수 vs GPT 분류 일치율 | `python -m eval.label_agreement --tag exp1b_casedesc --lab eval/grading/label_check2.csv` |

### ② 검색·예상 (리포트 4·6·7항)

| 스크립트 | 하는 일 | 실행 예 |
|---|---|---|
| `compare_embed.py` | 임베딩 2종 비교 (메모리에서만, store에 안 씀) | `python -m eval.compare_embed` |
| **`run_eval.py`** | **Hit@1·Hit@3·MRR + 의도·특징·등급** (chain 경유, `--mode search`는 검색기만) | `HF_HUB_OFFLINE=1 python -m eval.run_eval --tag run1_chain` |
| `holdout.py` | 숨겨 두기 검증 (사례 하나를 빼고 예상) | `HF_HUB_OFFLINE=1 python -m eval.holdout --tag day5_holdout_v1` |
| `repeat3.py` | LLM 판정 3회 반복 일치율 | `HF_HUB_OFFLINE=1 python -m eval.repeat3 --tag day5_repeat3_v1` |
| `feat_probe.py` | exp9 평가셋 밖 특징 확인 (3회) | `python -m eval.feat_probe --tag exp9_probe_after` |
| `synth_check.py` | exp10 종합 판정·출처 종류 섞임 확인 | `HF_HUB_OFFLINE=1 python -m eval.synth_check --tag exp10_check_after` |
| `inspect_fail.py` | 실패 문항의 top3 청크와 정답 청크 순위를 나란히 출력 (진단용, 조회만) | `HF_HUB_OFFLINE=1 python -m eval.inspect_fail --tag run1_chain` |

### ③ 답변 (리포트 6·7항)

| 스크립트 | 하는 일 | 실행 예 |
|---|---|---|
| **`check_answers.py`** | **답변 생성 → 사람이 채점할 md** (`report/evidence/<태그>.md`) | `HF_HUB_OFFLINE=1 python -m eval.check_answers --tag final_answers_v1` |
| `check_refusal.py` | 범위 판정 거절률·잘못 거절, 1위 유사도 분포 | `HF_HUB_OFFLINE=1 python -m eval.check_refusal --tag day4_scope_v3` |

### ④ 챗봇 기능 확인 (Day 4, 가이드 4-7)

| 스크립트 | 확인하는 것 |
|---|---|
| `check_stream.py` | 문장 단위 스트리밍 글 = 최종 본문, 첫 글자까지 시간 |
| `check_multiturn.py` | 이어지는 질문을 다시 쓴 결과에 필요한 말이 들어갔는지 |
| `check_filters.py` | 필터 없음 = 기본 검색, 필터를 고르면 근거가 모두 조건 안인지 |

실행 형식은 모두 `HF_HUB_OFFLINE=1 python -m eval.<이름> --tag <태그>`.

### ⑤ 심화: 기획서 확인 (리포트 10항)

| 스크립트 | 하는 일 | 실행 예 |
|---|---|---|
| `check_plan_gold.py` | 정답표 검사 (원문·anchor·허용값·등급 재계산, LLM 없음) | `python -m eval.check_plan_gold --gold eval/plans/plan_gold2.jsonl` |
| `plan_eval.py` | 측정 3종 (추출 재현율·칸 정답률·검색 A vs B, 3회) | `HF_HUB_OFFLINE=1 python -m eval.plan_eval --tag deep_plan2_after --gold eval/plans/plan_gold2.jsonl` |

### ⑥ 제출 전 구조 검사

| 스크립트 | 하는 일 | 실행 예 |
|---|---|---|
| `check_layers.py` | 계층 분리 · 같은 체인 · 개인정보 · API 키 → `report/evidence/check_layers.md` (`--live`는 OpenAI 1회) | `HF_HUB_OFFLINE=1 python -m eval.check_layers` |

## 4. 핵심 수치를 다시 내려면 (순서)

같은 태그로 돌리면 기존 결과를 덮어쓴다. 다시 잴 때는 `_v2`처럼 **새 태그**를 붙인다.

1. `run_eval --tag run1_chain` → 기준선 검색 지표
2. `check_answers --tag final_answers_v1` → 답변 원문 → `grading/final_grading.csv`로 사람 채점
3. `holdout --tag day5_holdout_v1` → 숨겨 두기
4. `label_agreement --tag exp1b_casedesc --lab eval/grading/label_check2.csv` → 최종 분류 일치율
5. `check_layers` → 구조 검사

LLM 단계(특징 추출·범위 판정·답변 문장)는 temperature 0에서도 실행마다 다를 수 있다. 그래서 2·3회 반복 결과(`repeat3`, `exp9_feat_prompt_r1~r3`)를 함께 본다.