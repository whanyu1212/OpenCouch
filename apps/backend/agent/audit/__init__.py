"""Safety-event capture and operator audit ledger helpers.

Runtime code captures only minimal structured safety events. Operator review,
summaries, exports, and retention work run later over the ledger and are never
loaded into working memory for response generation.
"""
