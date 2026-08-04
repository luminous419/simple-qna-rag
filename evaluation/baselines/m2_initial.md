# M2 최초 품질 기준선

상태: **사용자 승인 완료**  
승인일: 2026-08-05  
실행 시각(UTC): 2026-08-04 14:56:21  
실행 commit: `95a4fd17d6658e548658f4e922750ae114625851`  
Git dirty: `false`

## 기준선 목적

이 문서는 M2 완료 시점의 실제 Retrieval, live Routing 및 Answer 품질·성능을 고정한 최초 기준선이다. 수치가 목표를 충족한다는 선언이 아니라, 이후 M3 변경을 같은 dataset과 실행 환경에서 비교하기 위한 출발점이다.

최초 실행은 dataset 76건 전체를 대상으로 수행했고 validate, Retrieval, Routing, Answer 네 단계가 모두 성공했다. 사용자가 실행 결과와 주요 실패 양상을 검토한 뒤 이 기준선을 승인했다.

## 실행 식별 정보

| 항목 | 값 |
|---|---|
| Git commit | `95a4fd17d6658e548658f4e922750ae114625851` |
| Git dirty | `false` |
| Python | `3.11.8` |
| Dataset | `evaluation/datasets/golden.jsonl` |
| Dataset SHA-256 | `61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a` |
| Corpus manifest SHA-256 | `5c0d648d032bc231c93cf1c545e5b3f8d26337da56677a4a0c1a66f24c82374a` |
| `index.faiss` SHA-256 | `c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820` |
| `index.pkl` SHA-256 | `3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00` |
| Embedding model | `BAAI/bge-m3` |
| Reranker model | `BAAI/bge-reranker-v2-m3` |
| Ollama model | `gpt-oss:20b` |
| 전체 실행 시간 | 2,742.44초 (약 45분 42초) |

Retrieval과 Answer가 독립적으로 계산한 corpus manifest 및 vectorstore fingerprint는 모두 일치했다.

전체 corpus manifest 18개 파일의 source ID, 크기와 SHA-256은 [기계 판독용 기준선](m2_initial.json)에 포함돼 있다.

## Dataset 구성

| 구분 | 사례 수 |
|---|---:|
| 전체 | 76 |
| Document QA | 51 |
| Web search | 15 |
| Boundary | 3 |
| Unanswerable | 7 |
| Retrieval 평가 대상 | 42 |
| Answer 평가 대상 | 29 |
| Assertion 보유 사례 | 22 |
| Abstention 기대 사례 | 7 |

## Retrieval 기준선

평가 대상 42건이 모두 기술적으로 성공했고 34건은 `relevant_sources`가 없어 제외됐다.

| 지표 | 결과 |
|---|---:|
| Recall@1 | 94.05% |
| Recall@3 | 94.05% |
| Recall@5 | 95.24% |
| Recall@10 | 97.62% |
| MRR@10 | 98.21% |
| nDCG@10 | 95.43% |
| 평균 latency | 16.84초 |
| median latency | 16.85초 |
| p95 latency | 22.61초 |

단계별 평균 latency:

| 단계 | 평균 |
|---|---:|
| BM25 | 0.52ms |
| Dense | 115.35ms |
| RRF | 0.04ms |
| MMR | 14,349.31ms |
| Reranker | 2,377.09ms |

검색 품질은 높지만 MMR이 Retrieval 시간 대부분을 차지한다. M3에서 문서 임베딩 재사용 또는 MMR 계산 경로 최적화를 우선 검토한다.

## Routing 기준선

76건 모두 기술 오류 없이 평가됐다.

| 지표 | 결과 |
|---|---:|
| 전체 정확도 | 77.63% (59/76) |
| Document QA precision | 100.00% |
| Document QA recall | 72.13% |
| Document QA F1 | 83.81% |
| Web search precision | 46.88% |
| Web search recall | 100.00% |
| Web search F1 | 63.83% |
| 평균 latency | 5.44초 |
| median latency | 4.27초 |
| p95 latency | 12.94초 |

오분류 17건은 모두 Document QA를 Web search로 보낸 경우다. Web search 질문을 Document QA로 잘못 보낸 경우는 없다.

주요 패턴은 최신 연도, 정책, 경제, 부동산과 기업 동향을 언급하지만 실제로는 로컬 문서에 답이 있는 질문이다. 현재 라우터는 최신 정보 누락을 피하는 방향으로 편향돼 있으며 문서 질문을 웹으로 과다 라우팅한다.

## Answer 기준선

평가 대상 29건이 모두 기술적으로 성공했다.

| 지표 | 자동 평가 결과 |
|---|---:|
| Assertion 통과율 | 75.00% (24/32) |
| Abstention 정확도 | 89.66% (26/29) |
| Source any-hit | 100.00% |
| Source 평균 recall | 95.45% |
| Intent 정확도 | 51.72% (15/29) |
| 평균 End-to-End latency | 55.48초 |
| median latency | 54.16초 |
| p95 latency | 74.88초 |

### 사람 검토 해석

자동 assertion이 놓친 8개 핵심 사실은 모델 답변에 의미상 모두 포함돼 있었다. 다음과 같은 표면 표현 차이로 문자열 규칙이 false negative를 만들었다.

- `0.7%`와 `0.7 %`
- `1.0%포인트`와 `1.0 pp`
- `chunk size`와 `chunk_size`
- “추가”와 “병합·삽입”
- “통합”과 “결합”

자동 abstention false negative 3건도 실제로는 모두 “제공된 문서에 정보가 없다”고 올바르게 답했다. 현재 detector가 제한된 공식 문구만 인식하기 때문에 자동 점수가 실제 거절 품질을 과소평가했다.

Intent 정확도 51.72%는 실제 개선 대상이다. 특히 `yesno`가 comparison/explanation/other로, `uncertain`이 comparison/explanation/other로 분산되는 문제가 크다.

## M3 우선 개선 후보

1. MMR 문서 임베딩 재사용 또는 계산 경로 최적화
2. 최신 연도·정책·기업 관련 Document QA의 Web search 과다 라우팅 개선
3. `yesno`와 `uncertain` 중심의 intent classifier 개선 또는 구조 단순화
4. assertion 정규화와 abstention 표현 인식 확장
5. Answer 평균 55.48초, p95 74.88초의 End-to-End latency 개선

M3에서 설정이나 모델을 바꿀 때는 이 기준선과 같은 dataset을 사용하고 dataset/corpus/vectorstore fingerprint가 같은지 먼저 확인한다. fingerprint가 다르면 동일 조건 비교로 간주하지 않는다.

## 알려진 한계

- Assertion coverage는 핵심 문자열 포함 여부만 확인하며 답변 전체의 진실성, faithfulness 또는 문맥 왜곡을 보증하지 않는다.
- Abstention detector는 제한된 표현을 사용하므로 의미상 올바른 거절을 false negative로 분류할 수 있다.
- Intent 정확도는 classifier 라벨과 골든 라벨의 일치율이며 답변의 내용 정확성과 동일하지 않다.
- Corpus manifest와 vectorstore fingerprint는 동일 파일을 사용했는지는 보여 주지만, vectorstore가 현재 embedding model과 chunk 설정으로 생성됐다는 provenance는 보증하지 않는다. 이를 완전히 해결하려면 인덱스 생성 시 sidecar manifest를 기록해야 한다.
- 실제 설치된 Python dependency snapshot은 이 기준선에 포함되지 않았다. 동일한 Git commit과 fingerprint에서도 라이브러리 버전 차이가 남을 수 있다.
- 상세 timestamped report에는 질문과 모델 답변이 포함될 수 있으므로 Git에서 제외하며, 고정 기준선에는 집계 수치와 비민감 요약만 기록했다.

## 승인된 원본 실행

로컬 상세 산출물 식별 경로:

- `evaluation/reports/m2_full/baseline_20260804T145621362462Z.json`
- `evaluation/reports/m2_full/baseline_20260804T145621362462Z.md`
- `evaluation/reports/m2_full/retrieval/retrieval_20260804T142238259802Z.json`
- `evaluation/reports/m2_full/routing/routing_20260804T142931996640Z.json`
- `evaluation/reports/m2_full/answers/answers_20260804T145621300637Z.json`
- `evaluation/reports/m2_full/answers/answers_20260804T145621300637Z_worksheet.md`

`evaluation/reports/`는 Git에서 제외되므로 위 경로는 승인된 로컬 실행을 식별하기 위한 기록이다. 장기 비교에 필요한 값은 이 문서와 `m2_initial.json`에 고정했다.
