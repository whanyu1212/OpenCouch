"""Download the BHH crisis benchmark into `evals/data/`.

BHH ("Between Help and Harm", Arnaiz-Rodriguez et al., JMIR Mental Health 2026)
is CC-BY-NC-4.0, so it is downloaded into `evals/data/` (gitignored) and never
committed. Its labels are crisis *types*, not severity levels:

- validation: about 200 examples labelled by humans; ties dropped.
- test: about 2,000 examples labelled by gpt-4o-mini (majority of 3 runs), a
  weaker ground truth.

Each example is stored as a list of sentences, joined here into one message.

    uv run python -m evals.crisis.bhh
"""

import json
import time
from dataclasses import asdict
from typing import Any, Final

import httpx

from evals.crisis.cases import BHH_LABELS, BHH_META_PATH, DATA_DIR, Case, bhh_path
from evals.crisis.runs import DatasetInfo

DATASET: Final = "arnaiztech/llms-mental-health-crisis-benchmark"
DATASET_INFO_URL: Final = f"https://huggingface.co/api/datasets/{DATASET}"
ROWS_URL: Final = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE: Final = 100  # the datasets server's maximum
# Pause between pages so a full download stays under the rate limit.
PAGE_DELAY_SECONDS: Final = 0.5
MAX_ATTEMPTS: Final = 6
SPLITS: Final = ("validation", "test")
LABEL_CONFIGS: Final = {
    "validation": "human_merged_labels",
    "test": "llm_merged_labels",
}


def _get_with_retries(
    client: httpx.Client, url: str, params: dict[str, str | int]
) -> dict[str, Any]:
    # The datasets server returns the odd 502 and rate-limits a full download
    # (429). Both clear after a wait; any other error is real.
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.get(url, params=params)
        transient = response.status_code == 429 or response.status_code >= 500
        if not transient:
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            return body
        if attempt < MAX_ATTEMPTS:
            retry_after = response.headers.get("retry-after", "")
            time.sleep(int(retry_after) if retry_after.isdigit() else 2**attempt)
    raise RuntimeError(
        f"Gave up after {MAX_ATTEMPTS} attempts: HTTP {response.status_code} from {url}"
    )


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
        time.sleep(PAGE_DELAY_SECONDS)
        rows.extend(item["row"] for item in body["rows"])
        if len(rows) >= body["num_rows_total"] or not body["rows"]:
            return rows


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
        response = client.get(DATASET_INFO_URL)
        response.raise_for_status()
        revision = response.json()["sha"]
        for split in SPLITS:
            cases = fetch_split(client, split)
            with bhh_path(split).open("w") as out:
                out.writelines(json.dumps(asdict(case)) + "\n" for case in cases)
            print(f"{split}: {len(cases)} cases")
    info = DatasetInfo(dataset=DATASET, revision=revision, license="cc-by-nc-4.0")
    BHH_META_PATH.write_text(info.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
