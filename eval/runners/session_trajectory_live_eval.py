"""Runner for long-session trajectory evaluation with the real thread runtime."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agent.models import Channel
from agent.persistence import PersistentAgentRuntime
from core.config import create_configured_llm_client
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from services.llm.base import BaseLLMClient

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "session_trajectory_long_v1.json"
)
SQLITE_PATH = Path(__file__).resolve().parents[1] / ".session_trajectory_eval.sqlite3"
console = Console()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run long-session trajectory evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "deterministic", "hybrid"],
        default="auto",
        help=(
            "Evaluation mode. 'auto' uses the configured LLM client when available "
            "and falls back to deterministic mode otherwise."
        ),
    )
    parser.add_argument(
        "--case",
        default=None,
        help="Optional single case id to run.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress turn-by-turn logging and only print pass/fail summaries.",
    )
    return parser


def _load_cases(case_id: str | None = None) -> list[dict]:
    cases = json.loads(DATASET_PATH.read_text())
    if case_id is None:
        return cases
    return [case for case in cases if case["id"] == case_id]


def _resolve_llm_client(mode: str) -> tuple[BaseLLMClient | None, str]:
    if mode == "deterministic":
        return None, "deterministic"
    if mode == "hybrid":
        return create_configured_llm_client(), "hybrid"
    try:
        return create_configured_llm_client(), "hybrid"
    except Exception:
        return None, "deterministic"


def _check_state_expectations(expectation: dict, *, state: dict, output) -> list[str]:
    errors: list[str] = []

    if (
        "session_intent" in expectation
        and state.get("session_intent") != expectation["session_intent"]
    ):
        errors.append(
            f"expected session_intent={expectation['session_intent']}, got {state.get('session_intent')}"
        )
    if (
        "allowed_session_intents" in expectation
        and state.get("session_intent") not in expectation["allowed_session_intents"]
    ):
        errors.append(
            f"expected session_intent in {expectation['allowed_session_intents']}, got {state.get('session_intent')}"
        )
    if (
        "session_intent_source" in expectation
        and state.get("session_intent_source") != expectation["session_intent_source"]
    ):
        errors.append(
            f"expected session_intent_source={expectation['session_intent_source']}, got {state.get('session_intent_source')}"
        )
    if (
        "session_stage" in expectation
        and state.get("session_stage") != expectation["session_stage"]
    ):
        errors.append(
            f"expected session_stage={expectation['session_stage']}, got {state.get('session_stage')}"
        )
    if (
        "allowed_session_stages" in expectation
        and state.get("session_stage") not in expectation["allowed_session_stages"]
    ):
        errors.append(
            f"expected session_stage in {expectation['allowed_session_stages']}, got {state.get('session_stage')}"
        )
    if (
        "allowed_modes" in expectation
        and output.mode not in expectation["allowed_modes"]
    ):
        errors.append(
            f"expected mode in {expectation['allowed_modes']}, got {output.mode}"
        )
    if (
        "response_type" in expectation
        and output.response_type.value != expectation["response_type"]
    ):
        errors.append(
            f"expected response_type={expectation['response_type']}, got {output.response_type.value}"
        )
    if (
        "needs_clarification" in expectation
        and output.crisis.needs_clarification != expectation["needs_clarification"]
    ):
        errors.append(
            f"expected needs_clarification={expectation['needs_clarification']}, got {output.crisis.needs_clarification}"
        )
    if (
        "needs_crisis_response" in expectation
        and output.crisis.needs_crisis_response != expectation["needs_crisis_response"]
    ):
        errors.append(
            f"expected needs_crisis_response={expectation['needs_crisis_response']}, got {output.crisis.needs_crisis_response}"
        )

    return errors


def _check_final_text_expectations(
    expectation: dict, *, response_text: str
) -> list[str]:
    errors: list[str] = []
    lowered = response_text.lower()

    if "must_not_include_any" in expectation:
        for phrase in expectation["must_not_include_any"]:
            if phrase.lower() in lowered:
                errors.append(f"response included forbidden phrase: {phrase}")
    if (
        "max_question_marks" in expectation
        and response_text.count("?") > expectation["max_question_marks"]
    ):
        errors.append(
            f"response had {response_text.count('?')} question marks, expected at most {expectation['max_question_marks']}"
        )

    return errors


def _build_state_snapshot(state: dict, output) -> dict[str, Any]:
    """Extract the state fields that are most useful during eval inspection.

    Args:
        state: Final LangGraph state after the turn.
        output: Public agent output for the turn.

    Returns:
        A compact dict of inspectable session and safety metadata.
    """

    return {
        "mode": output.mode,
        "response_type": output.response_type.value,
        "session_intent": state.get("session_intent"),
        "session_intent_source": state.get("session_intent_source"),
        "session_stage": state.get("session_stage"),
        "session_stage_source": state.get("session_stage_source"),
        "session_stage_reason": state.get("session_stage_reason", ""),
        "current_goal": state.get("current_goal"),
        "open_loops": list(state.get("open_loops", [])),
        "needs_clarification": output.crisis.needs_clarification,
        "needs_crisis_response": output.crisis.needs_crisis_response,
        "crisis_level": output.crisis.level,
        "crisis_reason": output.crisis.reason,
    }


def _render_turn(
    case_id: str,
    turn_index: int,
    *,
    user_text: str,
    assistant_text: str,
    snapshot: dict,
    expectation: dict | None,
    errors: list[str],
) -> None:
    """Render one evaluated turn with conversation and state details.

    Args:
        case_id: Dataset case identifier.
        turn_index: One-based turn index.
        user_text: Current user message.
        assistant_text: Generated assistant message.
        snapshot: Inspectable state snapshot for the turn.
        expectation: Optional checkpoint expectation for the turn.
        errors: Checkpoint mismatches for the turn.
    """

    title = Text.assemble(
        ("Turn ", "bold white"),
        (str(turn_index), "bold cyan"),
        ("  ", ""),
        (case_id, "dim"),
    )
    body = Table(show_header=False, box=None, padding=(0, 1))
    body.add_column(style="cyan", no_wrap=True)
    body.add_column(style="white")
    body.add_row("user", user_text)
    body.add_row("assistant", assistant_text)
    body.add_row("mode", snapshot["mode"] or "-")
    body.add_row("type", snapshot["response_type"])
    body.add_row(
        "intent",
        f"{snapshot['session_intent'] or '-'} ({snapshot['session_intent_source'] or 'none'})",
    )
    body.add_row(
        "stage",
        f"{snapshot['session_stage'] or '-'} ({snapshot['session_stage_source'] or 'none'})",
    )
    body.add_row("stage reason", snapshot["session_stage_reason"] or "-")
    body.add_row("goal", snapshot["current_goal"] or "-")
    body.add_row(
        "open loops",
        " | ".join(snapshot["open_loops"]) if snapshot["open_loops"] else "-",
    )
    body.add_row(
        "safety",
        f"level={snapshot['crisis_level']} clarify={snapshot['needs_clarification']} crisis={snapshot['needs_crisis_response']}",
    )
    body.add_row("crisis reason", snapshot["crisis_reason"] or "-")
    if expectation is not None:
        body.add_row("checkpoint", json.dumps(expectation, ensure_ascii=True))
    if errors:
        body.add_row("result", "[red]FAIL[/red] " + " ; ".join(errors))
    else:
        body.add_row("result", "[green]PASS[/green]")

    border_style = "red" if errors else "blue"
    console.print(Panel(body, title=title, border_style=border_style))


def _render_case_header(case: dict) -> None:
    """Render the header for one session-trajectory case.

    Args:
        case: Dataset case definition.
    """

    console.print(Rule(f"[bold blue]{case['id']}[/bold blue]"))
    console.print(Panel(case["description"], border_style="blue"))


def _render_case_footer(case_id: str, failures: list[str]) -> None:
    """Render the final summary for one case.

    Args:
        case_id: Dataset case identifier.
        failures: Accumulated failures for the case.
    """

    if failures:
        console.print(
            Panel(
                "\n".join(failures),
                title=f"[bold red]FAIL {case_id}[/bold red]",
                border_style="red",
            )
        )
    else:
        console.print(Panel(f"PASS {case_id}", border_style="green"))


async def _evaluate_case(
    case: dict, *, llm_client: BaseLLMClient | None
) -> dict[str, Any]:
    thread_id = f"eval-{case['id']}-{uuid4().hex[:8]}"
    failures: list[str] = []
    turn_records: list[dict[str, Any]] = []

    async with PersistentAgentRuntime(SQLITE_PATH) as runtime:
        checkpoint_map = {
            checkpoint["turn"]: checkpoint["expect"]
            for checkpoint in case.get("checkpoints", [])
        }
        last_result = None

        for turn_index, turn in enumerate(case["turns"], start=1):
            last_result = await runtime.run_turn(
                thread_id=thread_id,
                message=turn["user"],
                channel=Channel.TEST,
                llm_client=llm_client,
            )
            expectation = checkpoint_map.get(turn_index)
            errors: list[str] = []
            if expectation is not None:
                errors = _check_state_expectations(
                    expectation,
                    state=last_result.state,
                    output=last_result.output,
                )
                for error in errors:
                    failures.append(
                        f"{case['id']} turn {turn_index}: {error} | user={turn['user']!r}"
                    )
            turn_records.append(
                {
                    "turn": turn_index,
                    "user": turn["user"],
                    "assistant": last_result.output.response_text,
                    "snapshot": _build_state_snapshot(
                        last_result.state, last_result.output
                    ),
                    "expectation": expectation,
                    "errors": errors,
                }
            )

        if last_result is None:
            return {
                "id": case["id"],
                "description": case["description"],
                "failures": [f"{case['id']}: no turns were executed"],
                "turn_records": [],
            }

        final_expectations = case.get("final_expectations", {})
        failures.extend(
            f"{case['id']} final: {error}"
            for error in _check_state_expectations(
                final_expectations,
                state=last_result.state,
                output=last_result.output,
            )
        )
        failures.extend(
            f"{case['id']} final: {error}"
            for error in _check_final_text_expectations(
                final_expectations,
                response_text=last_result.output.response_text,
            )
        )

        await runtime.reset_thread(thread_id)

    return {
        "id": case["id"],
        "description": case["description"],
        "failures": failures,
        "turn_records": turn_records,
    }


async def _run(mode: str, case_id: str | None, *, quiet: bool) -> int:
    cases = _load_cases(case_id=case_id)
    if not cases:
        console.print("[red]No matching session-trajectory eval cases found.[/red]")
        return 1

    llm_client, resolved_mode = _resolve_llm_client(mode)
    failures: list[str] = []

    console.print(
        Panel(
            f"Running session trajectory live eval in [bold cyan]{resolved_mode}[/bold cyan] mode on [bold]{len(cases)}[/bold] case(s).",
            border_style="blue",
        )
    )

    for case in cases:
        result = await _evaluate_case(case, llm_client=llm_client)
        case_failures = result["failures"]
        failures.extend(case_failures)
        if not quiet:
            _render_case_header(case)
            for record in result["turn_records"]:
                _render_turn(
                    case["id"],
                    record["turn"],
                    user_text=record["user"],
                    assistant_text=record["assistant"],
                    snapshot=record["snapshot"],
                    expectation=record["expectation"],
                    errors=record["errors"],
                )
            _render_case_footer(case["id"], case_failures)
        elif case_failures:
            console.print(f"[red]FAIL[/red] {case['id']}")
        else:
            console.print(f"[green]PASS[/green] {case['id']}")

    if failures:
        console.print(
            Panel(
                f"{len(failures)} session-trajectory expectation(s) failed.",
                border_style="red",
            )
        )
        return 1

    console.print(
        Panel(
            f"All {len(cases)} session-trajectory case(s) passed.",
            border_style="green",
        )
    )
    return 0


def main() -> int:
    """Run the long-session trajectory evaluation runner.

    Returns:
        Process exit code for the evaluation run.
    """

    args = _build_parser().parse_args()
    return asyncio.run(_run(args.mode, args.case, quiet=args.quiet))


if __name__ == "__main__":
    raise SystemExit(main())
