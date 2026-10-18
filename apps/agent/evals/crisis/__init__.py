"""Crisis detection evals: how well the user check catches risk (HY-211).

Jev's answers don't depend on our policy, so they are recorded once per
question-set version and re-scored offline as often as needed:

    uv run --env-file .env python -m evals.crisis.cases            # download BHH
    uv run --env-file .env python -m evals.crisis.record --split validation
    uv run python -m evals.crisis.score evals/results/crisis/<run>
"""
