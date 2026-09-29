# guardrail_agent

A CPU-only, Japanese-focused guardrail pipeline for LLM chat input — jailbreak/safety screening, PII redaction, and customer-harassment ("kasuhara") detection, plus an experimental typed-classification layer (Laya) for query triage.

It's meant to sit in front of an internal support chatbot and screen/sanitize user prompts before they reach the LLM.

## How it works

All logic lives in the `CPUHybridGuardrail` class in `app.py`, which chains four stages together in `process()`:

1. **Safety / jailbreak check** (`check_safety_ollama`) — sends the prompt to a local **Ollama** daemon running `llama-guard3:1b`. If it returns anything other than `"safe"`, the request is blocked. If Ollama is unreachable, the check **fails open** (`BYPASS_WARN`) rather than blocking — see [Known limitations](#known-limitations).
2. **Laya typed classification** (`classify_with_laya`) — uses the [Laya](https://huggingface.co/convaiinnovations/laya) `Router` to answer a battery of structured questions about the prompt in a single pass: `pii`, `department`, `intent`, `urgency`, `sensitivity`, `suggested_action`, `requires_auth`, `is_security_risk`. This is currently a work-in-progress: the result is computed and printed, but **not yet consumed** by `process()` — only the PII probability is checked internally, and no classification output affects the final pipeline decision yet.
3. **PII detection & redaction** (`mask_pii_presidio`) — uses Microsoft **Presidio** (`AnalyzerEngine` + `AnonymizerEngine`) with a spaCy `ja_core_news_trf` NLP engine, plus two custom regex recognizers for Japanese "My Number" (national ID) and Japanese phone numbers.
4. **Sentiment / harassment detection** (`check_sentiment_and_kasuhara`) — runs a Japanese BERT sentiment classifier (`cl-tohoku/bert-base-japanese-v3`) over the sanitized prompt and flags explicit customer-harassment keywords (責任者を出せ, 金返せ, 死ね, バカ, 訴えてやる, ボケ).

## Requirements

- Python 3.10+
- A local [Ollama](https://ollama.com) daemon with the `llama-guard3:1b` model pulled:
  ```bash
  ollama pull llama-guard3:1b
  ```
- The spaCy Japanese transformer model (not installable via `requirements.txt`):
  ```bash
  python -m spacy download ja_core_news_trf
  ```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download ja_core_news_trf
```

Then make sure Ollama is running locally with `llama-guard3:1b` pulled (see above).

## Usage

Run the three built-in sample cases (PII redaction, jailbreak attempt, customer harassment):

```bash
python app.py
```

Or run a single custom prompt:

```bash
python app.py "こんにちは。私のメールは user@example.com です。"
```

Each run prints the input and a result dict. On a passed check:

```python
{
    "status": "PASSED",
    "sanitized_prompt": "...",   # PII redacted
    "sentiment": {"label": ..., "score": ..., "kasuhara_detected": bool},
    "latency_ms": {"safety_check": ..., "pii_masking": ..., "sentiment": ..., "total_overhead": ...},
}
```

On a blocked (unsafe) prompt:

```python
{
    "status": "BLOCKED",
    "reason": "Safety Policy Violation (...)",
    "sanitized_prompt": "...",
    "latency_ms": {"safety_check": ...},
}
```

## Known limitations

This is an early-stage prototype:

- The Laya classification layer computes a rich set of triage signals (department, intent, urgency, sensitivity, suggested action, auth requirement, security risk), but only the `pii` flag currently affects behavior, and even that result is not wired into `process()`'s final decision.
- If the Ollama daemon is unreachable, the safety check **fails open** (passes the prompt through with a `BYPASS_WARN`) rather than blocking — this is a deliberate availability tradeoff worth revisiting before production use.
- Presidio's analyzer is hardcoded to `language="ja"`, so non-Japanese input is still analyzed as Japanese text.
- No automated tests yet.
