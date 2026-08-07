# 프로젝트 로드맵

## 비전

조직 또는 개인이 보유한 문서와 최신 웹 정보를 안전하고 검증 가능한 방식으로 연결하여, 로컬 환경에서도 신뢰할 수 있는 한국어 지식 도우미를 운영할 수 있게 합니다.

## 미션

- 문서 근거와 출처가 명확한 답변을 제공합니다.
- 검색 품질과 답변 품질을 직관이 아니라 지표로 개선합니다.
- 로컬 모델의 프라이버시 장점을 유지하면서 필요한 최신 정보만 웹에서 보완합니다.
- 장애 시 예측 가능한 폴백과 안전한 사용자 경험을 제공합니다.
- 작은 환경에서는 단순하게, 실제 규모가 필요할 때만 확장합니다.

## 제품 원칙

1. **근거 우선**: 답변보다 출처와 검증 가능성을 우선합니다.
2. **측정 후 최적화**: 평가 기준선 없이 모델이나 검색 알고리즘을 교체하지 않습니다.
3. **안전한 폴백**: 외부 서비스나 Agent가 실패해도 가능한 범위에서 문서 QA로 복구합니다.
4. **로컬 우선**: 문서와 핵심 모델은 기본적으로 로컬에서 처리합니다.
5. **필요 기반 확장**: 문서 수, 사용자 수, latency 지표가 요구할 때 확장 기술을 도입합니다.

## 현재 위치

```text
M0 Core RAG           완료
        |
M1 Agent & Safety     완료
        |
M2 Quality & CI       완료
        |
M2.5 Repo Structure   완료
        |
M3 Retrieval Quality  완료
        |
M4 Production Ready   예정                         <-- 현재 위치
        |
M5 Scale & Advanced   조건부
```

M2의 Phase 0~9를 완료했습니다. 76개 골든 사례, Retrieval·Routing·Answer evaluator, 통합 live baseline, 재현성 metadata와 Python/Node CI가 구현됐습니다. 최초 전체 live 실행은 사용자 승인을 거쳐 [M2 기준선](../evaluation/baselines/m2_initial.md)으로 고정됐습니다.

M2.5 Repository Restructuring의 Phase 0~5를 완료했습니다. 제품 package, 문서·테스트, Web·학습·모델 자산과 runtime 경로를 정리했고 PR #12의 Python/Frontend GitHub Actions 성공과 사용자 최종 승인을 확인했습니다.

M3 Retrieval & Domain Quality를 완료했습니다. 저장 벡터를 재사용해 Retrieval
평균 latency를 16.84초에서 2.21초로 줄이면서 검색 품질을 유지했고, 라우팅
정확도 중앙값은 75/76, 웹 recall은 모든 run에서 15/15를 달성했습니다. Answer
evaluator v2의 false negative를 줄였으며 paired blind 평가와 사용자 승인을
근거로 intent별 템플릿 대신 기본 템플릿을 채택했습니다. 다음 단계는 M4
Production Readiness입니다.

## 마일스톤

### M0 — Core RAG Foundation

**상태: 완료**

목표: 로컬 문서를 등록하고 검색 근거 답변을 생성하는 기본 시스템 구축

완료 범위:

- PDF/TXT 문서 등록과 FAISS 인덱스 생성
- BGE-M3 Dense Retrieval
- BM25 + Dense + RRF Hybrid Retrieval
- MMR와 Cross-Encoder 재정렬
- Intent Classifier와 의도별 프롬프트
- Ollama 기반 답변 생성
- Web UI, API, CLI

### M1 — Agent Routing, Reliability & Frontend Safety

**상태: 완료**

목표: 최신 정보와 문서 질문을 분리하고 장애 및 외부 콘텐츠를 안전하게 처리

완료 범위:

- LLM tool calling 기반 웹 검색/문서 QA 라우팅
- Agent 실패 시 키워드 라우터 폴백
- 웹 검색 실패 시 원본 질문으로 문서 QA 재시도
- DDGS 타임아웃 및 패키지 마이그레이션
- 웹 검색 결과 XSS 정화
- Vitest + jsdom 프런트엔드 보안 테스트
- marked/DOMPurify 로컬 vendor와 잠금 버전 동기화
- mock 기반 Agent 오케스트레이션 테스트

### M2 — Quality Baseline & Continuous Integration

**상태: 완료** — Phase 0~9 구현, live baseline 사용자 승인 및 CI 검증 완료

목표: 모든 후속 개선을 객관적으로 비교할 수 있는 품질·성능 기준선 구축

완료 산출물:

- 대표 질문과 관련 문서를 포함한 골든 평가셋
- Retrieval 지표: Recall@K, MRR, nDCG
- 답변 평가 기준: assertion coverage, abstention, 출처 일치, intent와 사람 검토 worksheet
- Agent 라우팅 정확도, confusion matrix와 오류 유형
- 검색 단계별 및 End-to-End latency 기준선
- Python/Node 정적 테스트 CI
- Ollama 라이브 테스트의 별도 실행 정책
- [승인된 최초 품질 기준선](../evaluation/baselines/m2_initial.md)
- [평가 패키지 사용 가이드](../evaluation/README.md)

완료 조건:

- 검색 설정 또는 모델 변경 전후를 동일한 명령으로 비교할 수 있습니다.
- Pull Request에서 외부 의존성이 없는 테스트가 자동 실행됩니다.
- 평가 결과와 허용 가능한 회귀 기준이 문서화됩니다.

개발 문서:

- [M2 요구사항](milestones/m2-quality-baseline/Requirement.md)
- [M2 개발 계획](milestones/m2-quality-baseline/Plan.md)

### M2.5 — Repository Restructuring

**상태: 완료** — Phase 0~5 구현, PR #12 GitHub Actions와 사용자 최종 승인 완료

목표: M2에서 확립한 동작과 기준선을 보존하면서 제품 코드, 테스트, 문서, 평가, 학습 자산과 runtime 데이터를 일관된 책임 경계로 재배치

확정 범위:

- `src/simple_qna_rag` 기반 Python package 구조
- 테스트, Web 자산, 학습·모델 자산과 문서의 역할별 분리
- `runtime/documents`, `runtime/vectorstore` 경로와 안전한 migration 계약
- current working directory에 의존하지 않는 경로 계산
- CLI entry point, CI, 문서 링크와 실행 명령 갱신
- M2 dataset, 승인 baseline과 fingerprint 보존 검증

완료 조건:

- 루트에는 프로젝트 진입 문서와 build/config 파일만 유지됩니다.
- 기존 Python/프런트엔드 테스트, Web/API와 CLI 동작이 새 구조에서 유지됩니다.
- M2 dataset·baseline hash와 corpus/vectorstore fingerprint가 보존됩니다.
- 기존 runtime 데이터가 삭제 또는 덮어쓰기 되지 않고 충돌 시 migration이 중단됩니다.
- clean install CI와 저장소 루트 외부 current directory 실행이 성공합니다.
- PR #12의 Python/Frontend CI 성공과 사용자 최종 승인을 완료했습니다.

개발 문서:

- [M2.5 상세 이전 계획](milestones/m2.5-repository-restructuring/Plan.md)
- [M2.5 Phase 0 기준 상태](milestones/m2.5-repository-restructuring/Phase_0_Baseline.md)
- [M2.5 Phase 1 결과](milestones/m2.5-repository-restructuring/Phase_1_Result.md)
- [M2.5 Phase 2 결과](milestones/m2.5-repository-restructuring/Phase_2_Result.md)
- [M2.5 Phase 3 결과](milestones/m2.5-repository-restructuring/Phase_3_Result.md)
- [M2.5 Phase 4 결과](milestones/m2.5-repository-restructuring/Phase_4_Result.md)
- [M2.5 Phase 5 최종 결과](milestones/m2.5-repository-restructuring/Phase_5_Final_Result.md)

### M3 — Retrieval & Domain Quality

**상태: 완료** — 구현, live 품질 gate 14/14 통과 및 사용자 승인 완료

목표: M2 기준선을 사용해 한국어 검색과 실제 도메인 답변 품질 개선

완료 범위:

- MMR 후보 문서의 저장 벡터 재사용과 안전한 폴백
- Retrieval 전후 비교, fingerprint 및 14개 통합 품질 gate
- precision-first 라우팅 신호 정책과 3회 반복 live 평가
- Answer evaluator v2 및 기존 결과 재채점
- Intent paired blind A/B와 기본 템플릿 구조 단순화
- 조건부 BM25 실험의 비진입 결정 기록

완료 조건:

- M2 기준선 대비 합의된 검색 및 답변 지표 개선
- latency 또는 자원 사용의 허용 범위 유지
- Intent Classifier 유지 여부와 근거 확정

완료 결과:

- [승인된 M3 품질 기준선](../evaluation/baselines/m3_initial.md)
- [M3 요구사항 추적표](milestones/m3-retrieval-domain-quality/Traceability.md)
- Retrieval 평균/p95 2.21초/2.40초, MMR 평균 8.38ms
- Routing 정확도 중앙값 75/76, Web recall 15/15(3회 전부)
- Answer source any-hit 100%, 평균 source recall 95.45%
- Answer 평균/p95 27.51초/37.34초

개발 문서:

- [M3 요구사항](milestones/m3-retrieval-domain-quality/Requirement.md)
- [M3 개발 계획](milestones/m3-retrieval-domain-quality/Plan.md)

### M4 — Production Readiness

**상태: 예정**

목표: 단일 사용자 로컬 데모를 넘어 반복 가능하고 관측 가능한 내부 서비스 운영

후보 범위:

- 구조화된 로그, 단계별 latency/error 메트릭
- 환경변수 기반 typed settings
- Python 의존성 잠금과 artifact 버전 관리
- blocking 작업과 동시 요청 처리 개선
- readiness/liveness 분리 및 부하 테스트
- 안전한 인덱스 생성, 버전 관리, 원자적 교체
- 인증, rate limiting, 입력 제한, reverse proxy 가이드
- 컨테이너와 배포 자동화

완료 조건:

- 새 환경에서 문서화된 절차로 동일한 서비스를 재현할 수 있습니다.
- 합의된 동시 요청, latency, 오류율 기준을 충족합니다.
- 장애 원인과 처리 단계를 로그 및 메트릭으로 확인할 수 있습니다.

### M5 — Scale & Advanced Capabilities

**상태: 조건부**

목표: 실제 사용 지표와 요구가 확인된 기능만 선택적으로 확장

검토 후보:

- IVF/HNSW 또는 외부 벡터 데이터베이스
- semantic cache와 분산 작업 처리
- 대화 히스토리와 참조 해석
- 복합 질문 decomposition과 multi-hop retrieval
- 표, 이미지, OCR을 포함한 멀티모달 문서
- 사용자 피드백 기반 active learning

착수 조건:

- 문서 수나 트래픽이 M4에서 정의한 용량 기준을 초과하거나
- 실제 사용자 피드백에서 해당 기능의 반복 수요가 확인되어야 합니다.

## 우선순위 요약

1. M2.5에서 저장소 구조와 경로 계약을 정리합니다.
2. 완료된 M2 기준선에서 M3의 개선 목표와 허용 회귀 범위를 합의합니다.
3. M3에서 기준선에 근거해 검색과 도메인 품질을 개선합니다.
4. M4에서 운영·배포·동시성 문제를 해결합니다.
5. M5는 실제 규모와 사용자 요구가 확인된 경우에만 진행합니다.

## 개발 계획 문서 정책

개별 마일스톤의 구체적인 개발 계획은 다음 형식으로 관리합니다.

```text
Development_{Milestone}_Plan.md
```

예시:

```text
docs/milestones/{milestone-slug}/Plan.md
```

개발 계획 문서는 다음 조건이 모두 충족된 마일스톤에 대해서만 작성합니다.

- 마일스톤 착수가 합의됨
- 작업 범위와 제외 범위가 확정됨
- 담당자 또는 의사결정 주체가 정해짐
- 세부 수용 기준과 검증 방법이 정해짐

Roadmap에 마일스톤이 존재한다는 이유만으로 개발 계획 파일을 미리 만들지 않습니다. 따라서 프로젝트 상황에 따라 `Development_{Milestone}_Plan.md`가 하나도 없을 수 있으며, 진행 과정에서 마일스톤별로 순차적으로 추가될 수 있습니다.

M2는 착수 범위와 수용 기준이 확정되어 다음 문서를 생성했습니다.

- [M2 Quality Baseline 요구사항](milestones/m2-quality-baseline/Requirement.md)
- [M2 Quality Baseline 개발 계획](milestones/m2-quality-baseline/Plan.md)

향후 마일스톤은 동일한 조건이 충족될 때만 각각의 계획과 요구사항 문서를 추가합니다.

## 문제 추적

현재 해결해야 할 구체적인 기술 문제는 [Problem.md](Problem.md)에서 관리합니다. 문제가 해결되면 이 문서에 완료 이력을 누적하지 않고 해당 항목을 제거하며, 상세 변경 이력은 Git 커밋과 Pull Request에 남깁니다.
