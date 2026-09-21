# asr-clinical-validator

A scoring guard for clinical text produced by speech-to-text and LLM summarisation, before a provider reads it.

Written fresh, stdlib only, no dependencies. `python clinical_validator.py` runs the self-check.

## The problem it solves

Two things go wrong between a microphone and a clinical note. ASR mishears, and an LLM asked to summarise the result will faithfully summarise the mistake — so the note reaches a provider asserting something that cannot be true of any patient: a deceased patient who is ambulatory, a benign lesion that metastasised, a temperature in mmHg.

The obvious fix is a rule that rejects the note on a string match. In practice that fails immediately. Transcripts are noisy, and clinicians constantly discuss findings they are ruling out — "afebrile, negative for fever" contains both halves of a contradiction and is a perfectly correct note. A boolean checker buries the reviewer in false positives and gets switched off within a week.

## The approach

Score, don't reject.

- **Severity weighting.** A contradiction contributes a weighted penalty rather than a veto. One CRITICAL sinks a transcript on its own; two MEDIUMs do not.
- **Negation and hedge suppression.** A hit is checked against the surrounding ±100 characters for negating or hypothetical phrasing (`denies`, `ruled out`, `history of`, `if`). Hedged discussion of a contradiction is not an assertion of one.
- **Credit for substance.** Length and clinical term density add confidence back, because a long, dense, otherwise sound transcript with one MEDIUM hit is a different risk from a short garbled one.
- **Routing, not blocking.** `validate()` returns a confidence score. The caller picks the threshold and sends only what falls below it to human review, so the guard tightens or loosens without a code change.

```python
from clinical_validator import validate

result = validate(transcript)
if not result.accepted(threshold=0.7):
    queue_for_human_review(transcript, result.findings)
```

## Scope

The rule set is deliberately small and readable — it demonstrates the mechanism, not a complete clinical ontology. A production deployment swaps the rule list for a terminology-backed source (ICD-10, UMLS) and tunes the weights against a labelled corpus of real reviewer decisions; the scoring and suppression logic is unchanged by that swap, which is the point of keeping the rules as data.
