# asr-clinical-validator

Scores clinical text coming out of speech-to-text and LLM summarisation, before a provider reads it.

Stdlib only, no dependencies. `python clinical_validator.py` runs the self-check.

```python
from clinical_validator import validate

result = validate(transcript)
if not result.accepted(threshold=0.7):
    queue_for_human_review(transcript, result.findings)
```

## Why the score

ASR mishears. Ask an LLM to summarise the result and it summarises the mistake, so a note can reach a provider claiming a deceased patient walked to the ward, or a temperature of 120 mmHg.

Reject on string match and you drown your reviewers. Clinicians write about findings they are excluding all day, and "afebrile, negative for fever" holds both halves of a contradiction while being a correct note. Your reviewers clear a few dozen of those and turn the checker off inside a week.

Every rule here costs confidence instead of failing the transcript.

A contradiction subtracts a penalty weighted by severity. One CRITICAL sinks a transcript on its own, two MEDIUMs leave it standing.

The checker then reads 100 characters either side of the hit for negating or hypothetical phrasing (`denies`, `ruled out`, `history of`, `if`). A clinician ruling something out has asserted nothing, so the hit drops.

Length and clinical term density add confidence back. A long dense transcript with one MEDIUM hit carries different risk from a short garbled one.

`validate()` hands back the score. You pick the threshold and send only what falls below it to human review, which lets you tighten the guard without a code change.

## Scope

Eleven rules, enough to show the mechanism. For a real deployment you swap the list for a terminology-backed source such as ICD-10 or UMLS, then tune the weights against your reviewers' decisions. Rules are data, so that swap leaves the scoring alone.
