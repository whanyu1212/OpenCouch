"""The cases a recording is made from, and the BHH benchmark download.

BHH ("Between Help and Harm", Arnaiz-Rodriguez et al., JMIR Mental Health 2026)
is CC-BY-NC-4.0, so it is downloaded into `evals/data/` (gitignored) and never
committed. Its labels are crisis *types*, not severity levels:

- validation: about 200 examples labelled by humans; ties dropped.
- test: about 2,000 examples labelled by gpt-4o-mini (majority of 3 runs), a
  weaker ground truth.

Each example is stored as a list of sentences, joined here into one message.

    uv run python -m evals.crisis.cases
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import httpx

DATA_DIR: Final = Path(__file__).parents[1] / "data"
DATASET: Final = "arnaiztech/llms-mental-health-crisis-benchmark"
ROWS_URL: Final = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE: Final = 100  # the datasets server's maximum
SPLITS: Final = ("validation", "test")
LABEL_CONFIGS: Final = {
    "validation": "human_merged_labels",
    "test": "llm_merged_labels",
}

BHH_LABELS: Final = (
    "suicidal_ideation",
    "self_harm",
    "violent_thoughts",
    "anxiety_crisis",
    "substance_abuse_or_withdrawal",
    "risk_taking_behaviors",
    "no_crisis",
)


@dataclass(frozen=True)
class Case:
    """One message to screen, with its label.

    Attributes:
        id: Stable case ID.
        text: The user message.
        label: A BHH crisis type, or an expected risk level for our own cases.
    """

    id: str
    text: str
    label: str


def load_cases(path: Path) -> list[Case]:
    """Read cases from a JSON Lines file."""
    with path.open() as lines:
        return [Case(**json.loads(line)) for line in lines if line.strip()]


def bhh_path(split: str) -> Path:
    """Where a downloaded BHH split lives."""
    return DATA_DIR / f"bhh-{split}.jsonl"


def _fetch_rows(client: httpx.Client, config: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    while True:
        params: dict[str, str | int] = {
            "dataset": DATASET,
            "config": config,
            "split": split,
            "offset": len(rows),
            "length": PAGE_SIZE,
        }
        body = _get_with_retries(client, ROWS_URL, params)
        time.sleep(0.5)
        rows.extend(item["row"] for item in body["rows"])
        if len(rows) >= body["num_rows_total"] or not body["rows"]:
            return rows


def _get_with_retries(
    client: httpx.Client, url: str, params: dict[str, str | int]
) -> dict[str, Any]:
    # The datasets server returns the odd 502, and rate-limits a full download
    # (429). Both clear after a wait.
    for attempt in range(6):
        response = client.get(url, params=params)
        if response.status_code != 429 and response.status_code < 500:
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            return body
        retry_after = response.headers.get("retry-after", "")
        time.sleep(int(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1))
    response.raise_for_status()
    raise AssertionError("unreachable")  # pragma: no cover


def fetch_split(client: httpx.Client, split: str) -> list[Case]:
    """Download one BHH split, keeping only examples with a resolved label."""
    examples = _fetch_rows(client, "examples", split)
    labels = {
        row["example_id"]: row
        for row in _fetch_rows(client, LABEL_CONFIGS[split], split)
    }
    cases = []
    for example in examples:
        label = labels.get(example["example_id"])
        if label and label["label_resolved"] and label["final_label"] in BHH_LABELS:
            text = " ".join(example["inputs"])
            cases.append(Case(example["example_id"], text, label["final_label"]))
    return cases


def main() -> None:
    """Download both BHH splits and record which dataset revision they came from."""
    DATA_DIR.mkdir(exist_ok=True)
    with httpx.Client(timeout=60) as client:
        info = client.get(f"https://huggingface.co/api/datasets/{DATASET}").json()
        for split in SPLITS:
            cases = fetch_split(client, split)
            with bhh_path(split).open("w") as out:
                out.writelines(json.dumps(asdict(case)) + "\n" for case in cases)
            print(f"{split}: {len(cases)} cases")
    meta = {"dataset": DATASET, "revision": info["sha"], "license": "cc-by-nc-4.0"}
    (DATA_DIR / "bhh.meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
