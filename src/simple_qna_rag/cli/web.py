"""Lightweight Web CLI that handles --help/--check-config before importing the
FastAPI app (Design.md §5)."""

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simple-qna-rag-web",
        description="Simple Q&A RAG FastAPI 서버를 실행합니다.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="바인딩 host")
    parser.add_argument("--port", type=int, default=8000, help="바인딩 port")
    parser.add_argument("--documents-dir", help="원본 문서 경로 override")
    parser.add_argument("--vectorstore-dir", help="vectorstore 경로 override")
    parser.add_argument("--model-dir", help="intent classifier 모델 경로 override")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="모델/엔진을 로드하지 않고 설정만 검증한 뒤 종료합니다(REQ-002.5).",
    )
    return parser


def _cli_overrides_from_args(args: argparse.Namespace) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if args.documents_dir:
        overrides["SIMPLE_QNA_RAG_DOCUMENTS_DIR"] = args.documents_dir
    if args.vectorstore_dir:
        overrides["SIMPLE_QNA_RAG_VECTORSTORE_DIR"] = args.vectorstore_dir
    if args.model_dir:
        overrides["SIMPLE_QNA_RAG_MODEL_DIR"] = args.model_dir
    return overrides


def _run_check_config(cli_overrides: dict[str, str]) -> int:
    """REQ-002.5: 외부 모델 없이 검증하고 secret/credential/prompt/절대 private
    path를 출력하지 않는다. 공개 정책은 `settings.check_config_payload()`가
    schema(`FieldSpec.annotation`)로부터 선언적으로 계산한다 — bool/int/float/
    `Literal` enum처럼 닫힌 도메인 필드만 값 그대로 출력하고, `Path`와 나머지
    모든 `str` 필드(URL의 userinfo credential, PROMPT_TEMPLATE 등 임의 사용자
    문자열 포함)는 non-reversible metadata로만 노출한다."""
    from simple_qna_rag.settings import Settings, SettingsError, check_config_payload

    try:
        settings = Settings.from_sources(cli_overrides=cli_overrides)
    except SettingsError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payload = check_config_payload(settings)

    print(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    args = build_parser().parse_args()
    cli_overrides = _cli_overrides_from_args(args)

    if args.check_config:
        sys.exit(_run_check_config(cli_overrides))

    from simple_qna_rag.web.server import start_server

    start_server(host=args.host, port=args.port, cli_overrides=cli_overrides)


if __name__ == "__main__":
    main()
