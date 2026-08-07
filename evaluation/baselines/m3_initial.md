# M3 승인 품질 기준선

상태: **사용자 승인 완료**  
승인일: 2026-08-08  
실행 시각(UTC): 2026-08-07 15:59:03  
통합 판정: **14/14 gate 통과**

## 결정

M3는 MMR 후보 문서의 저장 벡터 재사용, precision-first 라우팅 정책, Answer
evaluator v2, 그리고 intent 구조 단순화를 채택했다. Phase 4의 paired blind
평가에서 기본 템플릿이 20건, intent 템플릿이 2건 선호됐고 7건은 동률이었다.
사용자는 비식별 worksheet 29건을 직접 검토하여 기존 사례별 채점을 승인했고,
이 결과와 `ANSWER_TEMPLATE_MODE=default` 채택을 승인했다. 기존 classifier
artifact는 롤백 가능하도록 보존한다.

## 최종 지표

| 영역 | 지표 | M3 결과 |
|---|---|---:|
| Retrieval | 평균 / p95 | 2.213초 / 2.404초 |
| Retrieval | MMR 평균 | 8.38ms |
| Retrieval | Recall@5 / Recall@10 | 95.24% / 97.62% |
| Retrieval | MRR@10 / nDCG@10 | 98.21% / 95.43% |
| Routing | 정확도 중앙값(3회) | 75/76 (98.68%) |
| Routing | 문서 recall 중앙값 | 60/61 (98.36%) |
| Routing | 웹 recall(각 run) | 15/15, 15/15, 15/15 |
| Answer v2 | assertion pass rate | 27/32 (84.38%) |
| Answer v2 | abstention accuracy | 29/29 (100%) |
| Answer | source any-hit / mean recall | 100% / 95.45% |
| Answer | 평균 / p95 latency | 27.51초 / 37.34초 |
| Intent | default 모드 평가 | 비활성(`accuracy=null`, 29건 제외) |

모든 Retrieval 품질·성능, Routing, 출처 및 Answer latency 기준을 통과했다.
Retrieval dataset, corpus manifest와 vectorstore fingerprint는 M2 승인 기준선과
동일하다.

## 재현성 식별자

| 항목 | 값 |
|---|---|
| Dataset SHA-256 | `61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a` |
| Corpus manifest SHA-256 | `5c0d648d032bc231c93cf1c545e5b3f8d26337da56677a4a0c1a66f24c82374a` |
| `index.faiss` SHA-256 | `c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820` |
| `index.pkl` SHA-256 | `3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00` |
| Router prompt SHA-256 | `478ca30cdebaa94a23213cbc6ab7a89da45aaf746cb4549fa3307801abbddd3c` |

기계 판독용 값은 [m3_initial.json](m3_initial.json)에 고정했다. 상세 질문과
모델 답변이 포함된 timestamped 원본은 Git에서 제외되는
`evaluation/reports/m3/m3-final-approved/`에 보존한다.

## 알려진 한계

- 최종 실행은 승인 전 작업 트리에서 수행되어 `git_dirty=true`다. dataset,
  corpus, vectorstore와 evaluator metadata hash를 별도로 고정해 비교 경계를
  보완했다.
- `default` 모드의 intent 필드는 응답 계약 호환성을 위해 `other`를 반환하므로
  intent accuracy 자체는 더 이상 채택 판단 지표가 아니다.
- 자동 평가기는 전체 답변의 사실성이나 문맥 왜곡을 보증하지 않는다. 향후
  운영 표본의 사람 검토를 병행해야 한다.
