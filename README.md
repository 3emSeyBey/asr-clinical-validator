# asr-clinical-validator

Scores clinical text coming out of speech-to-text and LLM summarisation, before a provider reads it.

Stdlib only, no dependencies. `python clinical_validator.py` runs the self-check.

```python
from clinical_validator import validate

result = validate(transcript)
if not result.accepted(threshold=0.7):
    queue_for_human_review(transcript, result.findings)
```

## Why scoring and not rejecting

ASR mishears. An LLM asked to summarise the result will summarise the mistake faithfully, so a note can reach a provider asserting something untrue of any patient: a deceased patient who is ambulatory, a benign lesion that metastasised, a temperature in mmHg.

A rule that rejects on string match does not survive contact with real transcripts. Clinicians constantly write about findings they are excluding, and "afebrile, negative for fever" contains both halves of a contradiction while being a correct note. A boolean checker floods the review queue with those and gets turned off inside a week.

So every rule costs confidence instead of failing the transcript:

A contradiction subtracts a weighted penalty by severity. One CRITICAL sinks a transcript alone, two MEDIUMs do not.

A hit is checked against 100 characters either side for negating or hypothetical phrasing (`denies`, `ruled out`, `history of`, `if`). Hedged discussion of a contradiction is not an assertion of one.

Length and clinical term density add confidence back. A long dense transcript with one MEDIUM hit is a different risk from a short garbled one.

`validate()` returns the score. The caller picks the threshold and sends only what falls below it to human review, so the guard tightens without a code change.

## Scope

Eleven rules, enough to show the mechanism. A real deployment swaps the list for a terminology-backed source (ICD-10, UMLS) and tunes the weights against reviewer decisions. Rules are data, so that swap does not touch the scoring.
