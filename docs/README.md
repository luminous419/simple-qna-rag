# 프로젝트 문서

현재 상태와 앞으로의 작업은 다음 문서에서 시작합니다.

- [Roadmap](Roadmap.md): 비전, 마일스톤과 현재 위치
- [Problem](Problem.md): 현재 해결되지 않은 문제
- [Repository Structure](architecture/Repository_Structure.md): 디렉터리 책임과 파일 배치 규칙
- [M2 Quality Baseline](milestones/m2-quality-baseline/): 완료된 M2 요구사항·계획·설계
- [M2.5 Repository Restructuring 계획](milestones/m2.5-repository-restructuring/Plan.md)과 [최종 결과](milestones/m2.5-repository-restructuring/Phase_5_Final_Result.md)
- [M2 리뷰](reviews/m2-quality-baseline/): 설계 및 코드 리뷰 이력

## 관리 원칙

- 현재 프로젝트 소개와 실행 방법은 저장소 루트 `README.md`에서 관리합니다.
- 현재 목표와 우선순위는 `Roadmap.md`, 미해결 항목은 `Problem.md`에서 관리합니다.
- 마일스톤의 요구사항·계획·설계는 `milestones/<milestone>/`에 둡니다.
- 리뷰 결과는 구현 문서와 섞지 않고 `reviews/<milestone>/`에 둡니다.
- 완료된 문서를 삭제하지 않되 현재 지침과 역사적 기록을 명확히 구분합니다.
- 문서를 이동하거나 이름을 바꾸면 저장소 전체의 Markdown local link 검사를 수행합니다.
