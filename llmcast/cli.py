from __future__ import annotations

import argparse
import json
import sys

import httpx
from openai import OpenAI

from llmcast import Llmcast


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llmcast",
        description="LLM을 사용하여 데이터를 다른 스키마로 변환합니다.",
    )
    parser.add_argument("source", help="소스 데이터 JSON 문자열 또는 파일 경로 (- 이면 stdin)")
    parser.add_argument("target", help="타겟 스키마 예시 JSON 문자열 또는 파일 경로")
    parser.add_argument("--api-key", default=None, help="OpenAI API 키 (미지정 시 OPENAI_API_KEY 환경변수 사용)")
    parser.add_argument("--model", default="gpt-4o-mini", help="사용할 모델 (기본값: gpt-4o-mini)")
    parser.add_argument("--verbose", "-v", action="store_true", help="디버그 로그 출력")

    args = parser.parse_args()

    source = _load_json(args.source)
    target = _load_json(args.target)

    client_kwargs = {}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key

    http_client = httpx.Client(verify=False)
    client = OpenAI(**client_kwargs, http_client=http_client)
    mapper = Llmcast(client=client, model=args.model, verbose=args.verbose)

    result = mapper.convert(source, target)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _load_json(value: str) -> dict | list:
    """JSON 문자열, 파일 경로, 또는 stdin(-)에서 JSON을 로드합니다."""
    if value == "-":
        return json.load(sys.stdin)

    # 파일 경로 시도
    try:
        with open(value) as f:
            return json.load(f)
    except (FileNotFoundError, IsADirectoryError):
        pass

    # JSON 문자열로 파싱
    return json.loads(value)


if __name__ == "__main__":
    main()
