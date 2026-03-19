from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import httpx
from openai import OpenAI

from llmcast import Llmcast


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llmcast",
        description="LLM을 사용하여 데이터를 다른 스키마로 변환합니다.",
    )
    parser.add_argument("source", help="소스 데이터 JSON 문자열 또는 파일 경로 (- 이면 stdin, -r 시 디렉토리)")
    parser.add_argument("target", help="타겟 스키마 예시 JSON 문자열 또는 파일 경로")
    parser.add_argument("--api-key", default=None, help="OpenAI API 키 (미지정 시 OPENAI_API_KEY 환경변수 사용)")
    parser.add_argument("--model", default="gpt-4o-mini", help="사용할 모델 (기본값: gpt-4o-mini)")
    parser.add_argument("-o", "--output", default=None, help="출력 파일 또는 디렉토리 경로 (기본값: output.json, -r 시 output/)")
    parser.add_argument("-r", "--recursive", action="store_true", help="디렉토리 내 모든 JSON 파일을 변환")
    parser.add_argument("--verbose", "-v", action="store_true", help="디버그 로그 출력")

    args = parser.parse_args()

    client_kwargs = {}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key

    http_client = httpx.Client(verify=False)
    client = OpenAI(**client_kwargs, http_client=http_client)
    mapper = Llmcast(client=client, model=args.model, verbose=args.verbose)

    if args.recursive:
        _run_recursive(mapper, args)
    else:
        _run_single(mapper, args)


def _run_single(mapper: Llmcast, args: argparse.Namespace) -> None:
    output = args.output or "output.json"

    if os.path.exists(output):
        raise FileExistsError(f"Output file already exists: {output}")

    source = _load_json(args.source)
    target = _load_json(args.target)

    result = mapper.convert(source, target)
    _write_json(output, result)
    print(f"Output saved to {output}")


def _run_recursive(mapper: Llmcast, args: argparse.Namespace) -> None:
    source_dir = args.source
    output_dir = args.output or "output/"

    if not os.path.isdir(source_dir):
        raise NotADirectoryError(f"Recursive mode requires a directory as source: {source_dir}")

    target = _load_json(args.target)
    os.makedirs(output_dir, exist_ok=True)

    source_files = sorted(glob.glob(os.path.join(source_dir, "**/*.json"), recursive=True))
    if not source_files:
        print(f"No JSON files found in {source_dir}")
        return

    for source_path in source_files:
        rel_path = os.path.relpath(source_path, source_dir)
        output_path = os.path.join(output_dir, rel_path)

        if os.path.exists(output_path):
            raise FileExistsError(f"Output file already exists: {output_path}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        source_data = _load_json(source_path)
        result = mapper.convert(source_data, target)
        _write_json(output_path, result)
        print(f"Output saved to {output_path}")


def _load_json(value: str) -> dict | list:
    """JSON 문자열, 파일 경로, 또는 stdin(-)에서 JSON을 로드합니다."""
    if value == "-":
        return json.load(sys.stdin)

    if os.path.isdir(value):
        raise IsADirectoryError(f"Expected a file, got a directory: {value}")

    # 파일 경로 시도
    try:
        with open(value) as f:
            return json.load(f)
    except FileNotFoundError:
        pass

    # JSON 문자열로 파싱
    return json.loads(value)


def _write_json(path: str, data: dict | list) -> None:
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
