## Description
<!-- Provide a concise summary of the changes, the problem being solved, and the architectural approach. -->

## Type of Change
<!-- Please check the relevant option: -->
- [ ] Bug fix (non-breaking change resolving an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Refactor (structural or architectural change)
- [ ] Documentation update
- [ ] Security/Safety update

## Verification & Testing
<!-- OpenCouch requires strict validation of safety boundaries and routing logic. Check all that apply: -->
- [ ] **Unit & Integration Tests:** `uv run pytest tests/unit tests/integration` completed successfully.
- [ ] **Therapeutic Contract Eval:** `apps/backend/.venv/bin/python -m eval.runners.therapeutic_contract_eval --plain` executed and passed.
- [ ] **Static Analysis:** `pre-commit run --all-files` completed without warnings.
- [ ] **No Secrets:** Verified that no API keys or sensitive credentials are included in this PR.

## Context & Impact
<!-- Please include any relevant diagrams, configuration changes, database migrations, or new dependencies introduced. -->

## Related Issues
<!-- e.g., Resolves #123 -->
