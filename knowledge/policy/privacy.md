# Privacy Policy Notes

Prompt and response generation should assume:
- users may share sensitive mental health information
- only necessary context should be injected into prompts
- stored memory should be minimal and reviewable
- deleted user data should not linger in avoidable prompt context

Operational preference:
- least necessary context
- explicit user control
- clear distinction between transcript history and extracted memory

Memory quality preference:
- prefer typed memory over free-text memory when possible
- only store information that is stable, repeated, or explicitly stated
- inferred memory should require direct evidence in the current turn, not just a broad classifier label
- correction or change-of-state turns should suppress new writes unless the correction is explicitly modeled
- episodic recall should be selective and pattern-oriented, not a default retrieval path
- stored memory should remain inspectable, deduplicated, and easy to revise or delete later
