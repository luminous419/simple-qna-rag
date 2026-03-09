#!/usr/bin/env python3
"""
문서 질의 프로그램 (CLI 버전)

터미널 기반 대화형 인터페이스
RAG 코어 엔진을 사용
"""

import sys
from rag_engine import get_rag_engine


def is_exit_command(user_input: str) -> bool:
    """종료 명령어 확인"""
    exit_keywords = ['종료', '끝', 'stop', 'quit', 'exit', 'finish']
    return user_input.strip().lower() in exit_keywords


def format_source_documents(sources: list) -> str:
    """출처 문서 포맷팅"""
    if not sources:
        return "출처 정보 없음"

    formatted = []
    for source in sources:
        page_info = f" (페이지 {source['page']})" if source['page'] else ""
        formatted.append(f"  [{source['index']}] {source['source']}{page_info}")

    return "\n".join(formatted)


def run_query_loop(rag_engine):
    """사용자 질의를 받아 처리하는 메인 루프"""
    print("\n" + "=" * 60)
    print("💬 문서 Q&A 시스템에 오신 것을 환영합니다!")
    print("=" * 60)
    print("ℹ️  종료하려면 '종료', '끝', 'stop', 'quit', 'finish' 중 하나를 입력하세요.")
    print("=" * 60 + "\n")

    while True:
        try:
            # 사용자 입력 받기
            user_question = input("🤔 질문을 입력하세요: ").strip()

            # 빈 입력 처리
            if not user_question:
                print("⚠️  질문을 입력해주세요.\n")
                continue

            # 종료 명령어 확인
            if is_exit_command(user_question):
                print("\n👋 문서 Q&A 시스템을 종료합니다. 감사합니다!")
                break

            # 질의 처리
            print("\n🔍 검색 및 답변 생성 중...")

            # RAG 엔진 호출
            result = rag_engine.query(user_question)

            if result['success']:
                # 답변 출력
                print("\n" + "=" * 60)
                print("📝 답변:")
                print("=" * 60)
                print(result['answer'])

                # 출처 문서 출력
                if result['sources']:
                    print("\n" + "-" * 60)
                    print("📚 참고 문서:")
                    print("-" * 60)
                    print(format_source_documents(result['sources']))

                print("\n" + "=" * 60 + "\n")
            else:
                print(f"\n❌ 오류: {result['answer']}\n")

        except KeyboardInterrupt:
            print("\n\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            print("다시 시도해주세요.\n")


def main():
    """메인 함수"""
    print("=" * 60)
    print("📚 문서 질의 프로그램 시작 (CLI)")
    print("=" * 60)

    try:
        # RAG 엔진 초기화
        rag_engine = get_rag_engine()

        # 질의 루프 실행
        run_query_loop(rag_engine)

    except Exception as e:
        print(f"\n❌ 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
