"""Run crisis-resource lookup evals.

Default mode is deterministic fake-provider evaluation:

    apps/backend/.venv/bin/python eval/runners/run_crisis_resource_eval.py

Live provider mode is opt-in and loads the same `.env` files as the backend runtime:

    apps/backend/.venv/bin/python eval/runners/run_crisis_resource_eval.py --live --provider openai

The runner calls the framework-neutral crisis resource lookup service directly so
it can evaluate location extraction, search-call branching, fallback behavior,
and resource normalization without invoking the full agent runtime.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "apps" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import load_runtime_env  # noqa: E402
from agent.tools.grounded_search import (  # noqa: E402
    CrisisResourceLookupRequest,
    find_crisis_resources_for_request,
)
from llm.base import BaseLLMClient, StructuredResponseT  # noqa: E402
from llm.factory import LLMProvider, create_llm_client  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "eval" / "datasets" / "crisis_resources.jsonl"
VALID_STATUSES = {
    "found",
    "no_location",
    "location_refused",
    "no_verified_results",
}


@dataclass(slots=True)
class EvalCase:
    """One crisis-resource eval case loaded from JSONL."""

    id: str
    mode: str
    message: str
    transcript: tuple[Mapping[str, Any], ...]
    expected: dict[str, Any]
    fake_structured_responses: list[dict[str, Any] | Exception]


@dataclass(slots=True)
class EvalResult:
    """Serializable result for one eval case."""

    id: str
    mode: str
    passed: bool
    checks: list[str]
    failures: list[str]
    output: dict[str, Any]


class ScriptedCrisisResourceLLM(BaseLLMClient):
    """Fake client for deterministic crisis-resource eval cases."""

    def __init__(
        self,
        *,
        structured_responses: list[dict[str, Any] | Exception],
    ) -> None:
        self.structured_responses = list(structured_responses)
        self.calls: list[dict[str, Any]] = []

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "use_search": use_search,
                "response_schema": None,
            }
        )
        raise AssertionError("Crisis resource eval should use structured output.")

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "unused"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> StructuredResponseT:
        self.calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "use_search": use_search,
                "response_schema": response_schema.__name__,
            }
        )
        if not self.structured_responses:
            raise AssertionError("No fake structured response configured.")
        response = self.structured_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response_schema(**response)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to crisis resource JSONL dataset.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live-mode cases with a real provider client.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default="openai",
        help="Live provider to use when --live is set.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional live provider model override.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Run only the given case id. Can be provided multiple times.",
    )
    parser.add_argument(
        "--include-fake-with-live",
        action="store_true",
        help="When --live is set, also run fake cases.",
    )
    return parser.parse_args()


def _load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            fake_responses = [
                _decode_fake_response(response)
                for response in raw.get("fake_structured_responses", [])
            ]
            cases.append(
                EvalCase(
                    id=str(raw["id"]),
                    mode=str(raw.get("mode", "fake")),
                    message=str(raw["message"]),
                    transcript=tuple(raw.get("transcript", [])),
                    expected=dict(raw.get("expected", {})),
                    fake_structured_responses=fake_responses,
                )
            )
    return cases


def _decode_fake_response(response: dict[str, Any]) -> dict[str, Any] | Exception:
    error = response.get("__error__")
    if error:
        return RuntimeError(str(error))
    return dict(response)


def _select_cases(
    cases: list[EvalCase],
    *,
    live: bool,
    include_fake_with_live: bool,
    case_ids: list[str] | None,
) -> list[EvalCase]:
    selected = cases
    if case_ids:
        allowed = set(case_ids)
        selected = [case for case in selected if case.id in allowed]
    if live:
        return [
            case
            for case in selected
            if case.mode == "live" or (include_fake_with_live and case.mode == "fake")
        ]
    return [case for case in selected if case.mode == "fake"]


async def _run_case(
    case: EvalCase,
    *,
    live_client: BaseLLMClient | None,
) -> EvalResult:
    llm_client: BaseLLMClient
    fake_client: ScriptedCrisisResourceLLM | None = None
    if case.mode == "live":
        if live_client is None:
            raise RuntimeError("Live case selected without a live LLM client.")
        llm_client = live_client
    else:
        fake_client = ScriptedCrisisResourceLLM(
            structured_responses=case.fake_structured_responses
        )
        llm_client = fake_client

    failures: list[str] = []
    checks: list[str] = []

    try:
        location, resources, status = await find_crisis_resources_for_request(
            CrisisResourceLookupRequest(
                current_user_message=case.message,
                transcript=case.transcript,
            ),
            llm_client=llm_client,
        )
        output = {
            "location": location,
            "resources": resources,
            "status": status,
            "search_calls": _search_call_count(fake_client.calls)
            if fake_client is not None
            else None,
            "calls": fake_client.calls if fake_client is not None else [],
        }
    except Exception as exc:
        output = {"exception": repr(exc)}
        failures.append(f"raised exception: {exc!r}")
        return EvalResult(
            id=case.id,
            mode=case.mode,
            passed=False,
            checks=checks,
            failures=failures,
            output=output,
        )

    _score_expected(case.expected, output, checks, failures)
    return EvalResult(
        id=case.id,
        mode=case.mode,
        passed=not failures,
        checks=checks,
        failures=failures,
        output=output,
    )


def _search_call_count(calls: list[dict[str, Any]]) -> int:
    return sum(1 for call in calls if call.get("use_search") is True)


def _score_expected(
    expected: dict[str, Any],
    output: dict[str, Any],
    checks: list[str],
    failures: list[str],
) -> None:
    status = output["status"]
    location = output["location"]
    resources = output["resources"]

    if expected.get("valid_status"):
        if status in VALID_STATUSES:
            checks.append("status is valid")
        else:
            failures.append(f"status {status!r} is not one of {sorted(VALID_STATUSES)}")

    if "status" in expected:
        _check_equal(
            "status",
            actual=status,
            expected=expected["status"],
            checks=checks,
            failures=failures,
        )

    if "location" in expected:
        _check_equal(
            "location",
            actual=location,
            expected=expected["location"],
            checks=checks,
            failures=failures,
        )

    if "expected_search_calls" in expected and output.get("search_calls") is not None:
        _check_equal(
            "search_calls",
            actual=output["search_calls"],
            expected=expected["expected_search_calls"],
            checks=checks,
            failures=failures,
        )

    if expected.get("requires_resource"):
        if resources:
            checks.append("returned at least one resource")
        else:
            failures.append("expected at least one resource")

    if expected.get("requires_no_resources"):
        if not resources:
            checks.append("returned no resources")
        else:
            failures.append(f"expected no resources, got {resources!r}")

    if phone := expected.get("must_preserve_phone"):
        phones = [str(resource.get("phone", "")) for resource in resources]
        if phone in phones:
            checks.append(f"preserved phone {phone}")
        else:
            failures.append(f"expected phone {phone!r} in resources, got {phones!r}")

    if expected.get("if_found_requires_resource_fields") and status == "found":
        _check_found_resource_fields(resources, checks, failures)


def _check_equal(
    label: str,
    *,
    actual: Any,
    expected: Any,
    checks: list[str],
    failures: list[str],
) -> None:
    if actual == expected:
        checks.append(f"{label} matched {expected!r}")
    else:
        failures.append(f"{label} expected {expected!r}, got {actual!r}")


def _check_found_resource_fields(
    resources: list[dict[str, str]],
    checks: list[str],
    failures: list[str],
) -> None:
    if not resources:
        failures.append("status found requires at least one resource")
        return

    for index, resource in enumerate(resources):
        missing = [
            field
            for field in ("name", "phone", "url", "region")
            if not str(resource.get(field, "")).strip()
        ]
        if missing:
            failures.append(f"resource {index} missing fields: {missing}")
        else:
            checks.append(f"resource {index} has name, phone, url, and region")


def _clear_empty_provider_env_vars() -> None:
    """Treat empty provider API-key env vars as unset before loading dotenv files."""

    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.getenv(key) == "":
            os.environ.pop(key, None)


def _make_live_client(
    *,
    provider: str,
    model: str | None,
) -> BaseLLMClient:
    _clear_empty_provider_env_vars()
    load_runtime_env()
    return create_llm_client(provider=provider_as_literal(provider), model=model)


def provider_as_literal(provider: str) -> LLMProvider:
    if provider not in {"openai", "gemini"}:
        raise ValueError(f"Unsupported provider: {provider}")
    return provider  # type: ignore[return-value]


async def _amain() -> int:
    args = _parse_args()
    cases = _select_cases(
        _load_cases(args.dataset),
        live=args.live,
        include_fake_with_live=args.include_fake_with_live,
        case_ids=args.case_id,
    )
    if not cases:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": "No eval cases selected.",
                    "dataset": str(args.dataset),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    live_client = (
        _make_live_client(provider=args.provider, model=args.model)
        if args.live
        else None
    )
    results = [
        await _run_case(
            case,
            live_client=live_client,
        )
        for case in cases
    ]
    passed = sum(1 for result in results if result.passed)
    summary = {
        "passed": passed == len(results),
        "passed_count": passed,
        "failed_count": len(results) - passed,
        "total_count": len(results),
        "results": [
            {
                "id": result.id,
                "mode": result.mode,
                "passed": result.passed,
                "checks": result.checks,
                "failures": result.failures,
                "output": result.output,
            }
            for result in results
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
