# M3 Phase 5 — 조건부 한국어 BM25 tokenizer 실험: 미진입 기록

- 상태: **진입 조건 미충족 — 미진입**
- 관련 요구사항: M3-REQ-008, Requirement §3.2

## 진입 조건 (Plan.md §4 Phase 5, Design.md §9)

Phase 5 진입에는 다음 두 조건이 모두 필요하다.

1. Phase 2 이후 Retrieval 품질 floor가 안정적이다.
2. 사용자가 추가 실험 비용을 승인한다.

## 판정

- 조건 1은 충족됐다: Phase 2(`m3-p2a-stored-vector`, MMR 저장 벡터 재사용)가
  live 42건 실행에서 Requirement §4.1의 모든 Retrieval gate를 만족했고
  (Recall@1/3/5/10, MRR@10, nDCG@10이 M2 승인값과 완전히 동일), 채택되어
  `config.py`의 기본 경로가 됐다(`evaluation/reports/m3/m3-p2a-stored-vector/phase2_decision.md`).
- 조건 2는 충족되지 않았다: 이번 구현 세션은 별도 사용자 승인 절차 없이
  진행된 단일 구현 dispatch이며, "추가 실험 비용"(신규 tokenizer 후보 A/B,
  RSS peak 측정을 위한 반복 subprocess 실행 등)에 대한 명시적 승인을 받지
  않았다.

Requirement §3.2와 Plan.md는 "진입하지 않아도 M3는 완료할 수 있다"고 명시하므로,
이 미진입은 M3 완료의 결격 사유가 아니다.

## 결정

**Phase 5에 진입하지 않는다.** 다음을 지킨다.

- `src/simple_qna_rag/text_tokenizers.py`, `evaluation/experiments/bm25_tokenizer.py`
  등 Phase 5 전용 모듈을 만들지 않는다(M3-REQ-008 "미채택 실험 코드가
  production 의존성이나 기본 경로에 남아서는 안 된다"와 동일한 원칙을
  애초에 구현하지 않는 방식으로 지킨다).
- `config.py`의 `BM25_TOKENIZER` 기본값은 `"whitespace"`로 유지한다(§3.4).
  `_create_bm25_retriever()`는 변경하지 않는다 — 기존 M2 동작과 byte 단위로
  동일하다.

## 재개 조건

향후 마일스톤에서 진입하려면:

1. 사용자가 Phase 5 진입과 실험 비용(신규 tokenizer 후보 평가, 초기화
   시간·RSS peak 3회 반복 측정)을 명시적으로 승인한다.
2. Design.md §9.1~9.4의 tokenizer 경계·오프라인 A/B 하네스·채택 gate를
   그대로 구현한다.
3. Requirement §3.2의 네 조건(Recall@10 또는 nDCG@10 ≥ +1.00%p, 모든
   Retrieval floor 유지, 신규 필수 native runtime/외부 서비스/배포 크기
   급증 없음, 초기화 시간·메모리 증가 각각 ≤ 20%)을 모두 만족해야만
   production 기본값으로 제안한다.
