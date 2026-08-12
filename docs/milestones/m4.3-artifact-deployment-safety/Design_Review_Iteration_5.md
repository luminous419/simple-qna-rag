# M4.3 Artifact & Deployment Safety 설계 리뷰 — Iteration 5

리뷰 일자: 2026-08-12 (Asia/Seoul)  
리뷰어: Fresh Codex (독립 설계 reviewer)  
대상: Iteration 5 개정 [Design.md](Design.md), [Requirement.md](Requirement.md),
[Plan.md](Plan.md), [Traceability.md](Traceability.md),
[Design Review Iteration 1](Design_Review_Iteration_1.md),
[Iteration 2](Design_Review_Iteration_2.md),
[Iteration 3](Design_Review_Iteration_3.md),
[Iteration 4](Design_Review_Iteration_4.md),
[개발 orchestration guide](../../../milestone_dev_orchestration_guide.md), 현재
code/tests/[workflow](../../../.github/workflows/ci.yml), M4.1/M4.2 승인·예외 산출물  
판정 범위: pre-merge Design Quality Gate, 조건부 연장 Iteration 5

## 판정

**FAIL / STOP — 9.3/10**

CRITICAL 0건, MAJOR 1건, MINOR 0건이다. PASS 조건인 CRITICAL=0,
MAJOR=0, score 9.7/10 이상을 충족하지 못한다. DR-I4-MAJ-01과
DR-I4-MIN-03은 설계 수준에서 닫혔지만, DR-I4-MAJ-02와 같은 근본인
producer payload exact-identity 검증이 원본 receipt의 duplicate/schema/job
binding에서 다시 열려 있다.

[orchestration guide](../../../milestone_dev_orchestration_guide.md)의 조건부
연장 stop rule은 동일 근본 문제가 2회 연속 재발하면 남은 횟수와 관계없이
즉시 중단하도록 요구한다. Iteration 4의 DR-I4-MAJ-02와 이번
DR-I5-MAJ-01은 모두 “producer receipt가 선언한 payload identity를 exact하게
검증하지 않고 축약된 dict/set 표현만 신뢰한다”는 동일 근본 문제다. 따라서
Iteration 6로 자동 연장하거나 구현 Phase로 진입하면 안 된다. 재개하려면 아래
문제를 receipt parser의 exact tagged schema와 duplicate/job-binding negative oracle로
수정한 뒤, 사용자가 중단 이후 재개를 명시적으로 결정해야 한다.

Native Linux/Ollama/DDGS, protected M3 live, M4.1 live 14-gate는 실행하지
않았다. M4.1 operational은 계속 `BLOCKED`, protected M3 live는 계속
`NOT_RUN`, `operational_status=BLOCKED`, `overall_release_ready=false`다.
PASS receipt를 합성하지 않았고 self-hosted runner/environment approval을
변경하지 않았다.

## Findings

### DR-I5-MAJ-01 — payload exact filename/manifest 결합이 duplicate receipt entry와 producer identity substitution을 허용한다

- **Severity:** MAJOR
- **Iteration 4 closure:** DR-I4-MAJ-02 동일 근본 문제 재발
- **Requirement:** M4.3-REQ-004.3, M4.3-REQ-007.1/.3/.5,
  M4.3-REQ-008.1, M4.3-NFR-003/.005
- **근거:** [Design.md](Design.md)의 §8.2-b `_verify_payloads`는
  `payloads` entry를 `declared[entry["filename"]] = entry`로 dict에 넣는다.
  같은 filename이 두 번 선언되면 마지막 entry가 앞 entry를 조용히 덮어쓰므로
  `set(required_files) == set(declared)`가 통과한다. 이어지는
  `_evaluate_producer`도 `{p["filename"]: p["sha256"] for p in
  doc.get("payloads", [])}`로 다시 축약하므로 duplicate는
  `payload_hashes`와 canonical `payload_manifest_sha256`에서 완전히 사라진다.
  예를 들어 `container` receipt에 유효한 `layer_scan.json` entry를 두 번 넣고
  마지막 것과 실제 파일만 일치시키면, 원본 receipt의 filename multiset은 exact하지
  않지만 assembler와 baseline checker 모두 정상 두-filename dict만 관찰해 `OK`를
  낼 수 있다.
- **추가 근거:** 같은 §8.2-b `_check_identity`는 required key가 **포함**됐는지만
  검사하며 exact key set을 요구하지 않고, `schema ==
  "m43-producer-receipt-v1"`, `doc["job"] == 현재 평가 중 job`,
  `semantic_status`의 type/enum도 검사하지 않는다. 따라서 unknown key를 가진
  receipt나 `job="m43-deterministic"`이라고 자칭하는 receipt를 `container`
  evidence slot에 넣어도 payload 내용만 container spec에 맞추면 identity 검사를
  통과한다. producer별 exact filename set과 canonical manifest hash를 도입했지만,
  그 입력인 producer receipt 자체가 exact schema/job identity로 고정되지 않아
  output→receipt→baseline candidate 결합이 완전하지 않다.
- **구현 가능성 영향:** 문서의 pseudocode에는 malformed-entry guard가 동일한
  `if` 문으로 연속 두 번 나타나 첫 번째 `if`의 body가 없는 형태이므로 그대로는
  Python으로 구현할 수도 없다. 단순 편집 오탈자이지만 위 duplicate collapse와 같은
  parser 구간에 있어 검증된 executable contract라고 볼 수 없다.
- **필수 수정:** producer receipt에 status와 무관한 단일 exact top-level schema를
  적용하고 `schema` literal, `job == expected job`, identity field exact type/value,
  `semantic_status` enum을 검사한다. `payloads`는 entry exact-key/type/schema를 먼저
  검사하고 filename list 길이와 unique filename set 길이가 같음을 요구한 뒤에만
  dict로 변환해야 한다. `payload_manifest_sha256`은 이 검증된 unique ordered-or-
  canonical mapping에서 producer/assembler/checker가 각각 재계산해야 한다.
  같은 filename duplicate(동일 entry/서로 다른 hash 각각), unknown receipt key,
  wrong schema, receipt `job` swap, malformed payload entry를 assembler 동일 parser가
  `PAYLOAD_INVALID` 또는 typed identity failure로 거부하는 negative case를 추가한다.
  기존 same-count substitution, extra+omission 상쇄, cross-job filename set swap,
  malformed/mismatched manifest-hash 5개 baseline checker oracle도 그대로 보존한다.
- **재현 명령(구현 후):** `python -m pytest -q
  tests/unit/test_assemble_m4_evidence.py
  tests/unit/test_check_m4_baseline.py`에서 위 duplicate/schema/job substitution
  fixture가 exit 0인 테스트 suite 안에서 typed rejection으로 확인돼야 한다.
- **Gate:** pre-merge Design Quality Gate 차단. stop rule 때문에 이 리뷰 뒤 자동
  개선 iteration 또는 구현 진입 금지.

## Iteration 4 closure 재검증

| Iteration 4 finding | Iteration 5 판정 | mechanism/test-oracle 결론 |
|---|---|---|
| DR-I4-MAJ-01 history previous 대수 | **CLOSED** | `_read_previous_from_history`가 latest committed record의 `pre_pointer`를 반환하고 `latest.post_pointer == current`를 검증한다. `_read_history_rows`는 exact key/type/schema, filename↔body op_id, operation enum, unique contiguous `0..N-1` sequence를 fail-closed 검사한다. empty/empty+current/first `A→B`/second `B→C`/rollback `C→B`/duplicate/gap/filename mismatch/enum/current mismatch와 세 crash window 재사용을 합친 11-case+crash oracle이 명시됐다. |
| DR-I4-MAJ-02 producer payload identity | **OPEN / SAME-ROOT RECURRENCE** | job별 exact filename set과 canonical payload-manifest hash의 assembler 재계산 및 checker 재검산은 추가됐다. 그러나 원본 receipt의 duplicate filename이 dict 변환에서 소실되고 receipt exact schema/job binding이 없어 output→receipt→baseline 결합의 입력 identity가 exact하지 않다(DR-I5-MAJ-01). |
| DR-I4-MIN-03 pinned M3 hash | **CLOSED** | 실제 승인 hash `c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820`/`3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00`가 상수로 고정됐다. tracked `m3_initial.json`의 현재 값과 일치하며 `git diff --exit-code -- evaluation/baselines/m3_initial.*` byte-불변 gate도 유지됐다. |

## 전체 설계·추적성 재검토

- trust-before-pickle은 동일 dirfd에서 검증한 bytes를 재오픈 없이
  `deserialize_index`/`pickle.loads`에 전달하고 invalid `current`를 legacy로
  downgrade하지 않는 구조다. 이 경계는 유지됐다.
- same-filesystem staging, full-write/file-fsync/directory-fsync, immutable publish,
  atomic current replace, transition reconcile, rollback primitive 재사용,
  fd-relative retention/cleanup은 구현 가능한 순서와 실패 대수를 갖는다. history
  reader closure도 rollback/previous 보호를 복구했다.
- OCI production allowlist, non-root/read-only rootfs/tmpfs/drop-all/
  no-new-privileges, 모든 layer traversal/whiteout scan, production test seam 물리적
  제외 및 mock harness, digest-pinned image/index rollback runbook은 Requirement와
  추적된다. 이 리뷰에서는 Docker/native Linux/Ollama를 실행하지 않았다.
- 단일 `.github/workflows/ci.yml` hosted DAG, same run/attempt/SHA/event binding,
  protected `m3-live-regression-gate`의 trusted trigger, `[self-hosted, ollama-m3]`,
  `m3-live-regression` environment 불변 방향은 맞다. 현재 code/workflow는 아직
  M4.3 구현 전 상태이며 lifecycle/assembler/container/runbook 테스트 파일도
  계획 경로이므로 설계 문구를 실행 PASS로 승격하지 않았다.
- M4.1 예외는 merge `fd14eec`의 operational 미완료를 명시하고 M4.2 최종 검증은
  deterministic 범위만 승인한다. 현재 HEAD `648e3ab`과 Traceability도
  `M4.1_BLOCKED=true`, protected live `NOT_RUN`, `overall_release_ready=false`를
  일관되게 보존한다.

## 중단 원인, 잔여 문제, 재개 조건

중단 원인은 조건부 연장 Iteration 4와 5에서 producer payload identity의 exactness가
같은 축약(dict/set) 경계 때문에 연속 재발한 것이다. 잔여 문제는
DR-I5-MAJ-01 한 건이며 범위는 구체적이지만, guide의 조기 stop rule은 해결 가능성과
별개로 자동 Iteration 6을 금지한다.

재개 조건은 (1) coordinator/user가 중단 이후 재개를 명시적으로 결정하고,
(2) receipt exact schema/job binding과 duplicate-filename rejection pseudocode를
수정하며, (3) duplicate/schema/job substitution negative oracle를 기존 substitution/
swap/malformed manifest cases에 추가하고, (4) 새 fresh independent review가
CRITICAL/MAJOR 0, score 9.7 이상을 확인하는 것이다. 그 전에는 구현, workflow 변경,
commit/push/PR/merge로 진행하면 안 된다.

## 검증

- 읽기 전용 확인: 현재 Git history/working tree, M3 baseline fingerprint와 bytes,
  M4.1 operational exception, M4.2 final deterministic report, current direct-save/
  dangerous-load 경계, protected workflow block
- 문서 링크 검사: `python scripts/check_markdown_links.py`
- whitespace/patch 검사: `git diff --check`
- 실행하지 않음: Native Linux/Ollama/DDGS, M3 protected live, M4.1 live 14-gate,
  self-hosted runner/environment 승인 변경

