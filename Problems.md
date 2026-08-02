# 문제점 및 해결 현황

Agent 기반 웹검색/문서 QA 라우팅과 프런트엔드 보안 변경의 최신 재검토 결과입니다.

## 검증 결과

- 검토 기준: `HEAD` 이후 작업 트리 변경
- Python 테스트: `21 passed, 1 skipped`
- 프런트엔드 테스트: `9 passed`
- 정적 검사: `git diff --check` 통과
- vendor 일치 검사:
  - `static/vendor/marked.umd.js`가 설치된 잠금 버전과 바이트 단위로 일치
  - `static/vendor/purify.min.js`가 설치된 잠금 버전과 바이트 단위로 일치
- 스킵 항목: 실제 Ollama가 필요한 라이브 라우팅 테스트

## 현재 미해결 사항

현재 검토 범위에서 배포를 차단하거나 별도 수정이 필요한 새 문제는 발견되지 않았습니다.

실제 Ollama 모델의 라우팅 정확도를 검사하는 라이브 테스트는 기본 테스트 실행에서 제외됩니다. 이는 의도된 테스트 구성으로, 모델이나 라우팅 프롬프트를 변경할 때 다음 명령으로 별도 검증해야 합니다.

```bash
RUN_LIVE_LLM_TESTS=1 pytest test_agent_routing.py -v
```

## 해결된 사항

### 1. 운영 CDN과 테스트 의존성의 버전 불일치 가능성

**상태: 해결됨**

운영 페이지가 버전이 유동적인 CDN 대신 저장소의 로컬 정적 자산을 사용하도록 변경됐습니다.

- `marked`와 DOMPurify를 `static/vendor/`에서 제공합니다.
- `package-lock.json`이 테스트 의존성 버전을 고정합니다.
- `scripts/sync-vendor.js`가 설치된 npm 패키지의 배포 파일을 `static/vendor/`로 복사합니다.
- `postinstall`과 `npm run sync-vendor`로 vendor 파일을 갱신할 수 있습니다.
- 필수 라이브러리 로드에 실패하면 검색 UI를 비활성화하고 사용자에게 오류를 표시합니다.
- 현재 vendor 파일과 설치된 잠금 버전의 파일이 바이트 단위로 일치함을 확인했습니다.

#### 관련 위치

- `templates/index.html:7-11`
- `static/vendor/`
- `scripts/sync-vendor.js`
- `static/app.js:22-50`
- `package.json:6-16`
- `package-lock.json`

### 2. 브라우저 XSS 방어 회귀 테스트 부재

**상태: 해결됨**

렌더링 로직을 `static/render.js`의 테스트 가능한 함수로 분리하고 Vitest + jsdom 기반 회귀 테스트를 추가했습니다.

다음 공격 및 정상 동작을 자동 검증합니다.

- `<script>` 제거
- `onerror`, `onmouseover` 등 이벤트 핸들러 제거
- `javascript:` 및 `data:` 링크 차단
- 검색 결과 출처를 HTML이 아닌 텍스트로 렌더링
- 오류 메시지를 텍스트로 렌더링
- 정상 HTTPS 링크 유지 및 `noopener noreferrer` 적용
- 출처 더보기 표시 동작

#### 관련 위치

- `static/render.js`
- `frontend_tests/render.test.js`
- `package.json`
- `vitest.config.js`

### 3. 웹 검색 결과를 통한 XSS 위험

**상태: 해결됨**

- Markdown 변환 결과를 DOMPurify로 정화합니다.
- 허용 URL을 HTTP/HTTPS 프로토콜로 제한합니다.
- 출처 목록은 DOM 요소로 생성하고 외부 값을 `textContent`로 설정합니다.
- 오류 메시지도 안전한 DOM API로 렌더링합니다.
- 정상 외부 링크에 `target="_blank"`와 `rel="noopener noreferrer"`를 적용합니다.

#### 관련 위치

- `static/render.js:8-30`
- `static/render.js:33-45`
- `static/render.js:47-80`
- `static/app.js:110-115`

### 4. Agent 장애 경로의 문서 QA 폴백 누락

**상태: 해결됨**

Agent 호출 실패 또는 도구 미선택 후 키워드 라우터의 웹 검색까지 실패하면 원본 질문으로 문서 QA를 재시도합니다.

다음 장애 경로의 테스트도 통과합니다.

- Agent 예외 + 키워드 웹 검색 실패
- Agent 도구 미선택 + 키워드 웹 검색 실패
- 키워드 라우터의 직접적인 웹 검색 실패

#### 관련 위치

- `query_router.py:73-80`
- `test_agent.py:108-164`
- `test_query_router.py:73-96`

### 5. 웹 검색 타임아웃 설정 미적용

**상태: 해결됨**

`WEB_SEARCH_TIMEOUT`이 `DDGS(timeout=WEB_SEARCH_TIMEOUT)`에 전달되어 실제 HTTP 요청에 적용됩니다.

#### 관련 위치

- `config.py:98`
- `web_search.py:45-54`

### 6. DuckDuckGo 검색 패키지 마이그레이션

**상태: 해결됨**

사용 중단 경고가 발생하던 `duckduckgo-search` 대신 `ddgs>=9.0.0`을 사용합니다.

#### 관련 위치

- `requirements.txt`
- `web_search.py:10`

## 현재 결론

기존에 확인된 기능·보안·테스트·의존성 재현성 문제는 모두 해결됐습니다. 현재 자동 테스트 범위에서는 새 회귀가 발견되지 않았으며, 모델이나 라우팅 프롬프트 변경 시 라이브 Ollama 테스트를 별도로 실행하는 운영 절차만 유지하면 됩니다.
