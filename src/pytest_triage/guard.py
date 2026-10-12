"""Guardrails. Issue text is written by strangers and goes straight into prompts.

Defence in layers - no single one is trusted to hold:

  1. REDACT   secrets and personal paths are removed before text leaves the machine
  2. FENCE    issue text is wrapped in <issue> tags it cannot close or forge
  3. SIGNAL   known injection phrasings are flagged in the report for a human
  4. CONSTRAIN outputs are enums (labels, missing-info items, candidate numbers),
              re-validated in code; free text (reasons) is length-capped and stripped
              of links and @mentions
  5. TEMPLATE the reply draft is built from structured fields, never from model
              prose, so injected text can't flow through to a maintainer verbatim
  6. NEVER WRITE nothing here can post to GitHub; a human copies the draft or doesn't
"""

import re

SECRET_PATTERNS = [
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "<github-token>"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "<github-token>"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"), "<google-key>"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), "<api-key>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<aws-key>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
]
# Usernames inside home-directory paths in tracebacks: /Users/alice/..., C:\Users\alice\...
HOME_PATHS = re.compile(r"(?P<pre>/Users/|/home/|[A-Za-z]:\\\\?Users\\\\?)(?P<user>[^/\\\s]+)")

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |the )?(previous|prior|above)",
    r"(reveal|print|show) (your|the) (system )?prompt",
    r"you are now\b",
    r"\bsystem prompt\b",
    r"(label|mark|classify|close) (this|the issue) as\b",
    r"as an ai( language model)?\b",
    r"</?\s*issue\b",
    # Added after the 19-attack suite (2026-10-12): authority claims and orders aimed at
    # the bot landed 2 attacks. Each flags at most 2 of 6,416 real pytest issues.
    r"\btriage (bot|tool|tools|assistant)\b",
    r"\bmaintainer'?s? note\b",
    r"\balready triaged\b",
    r"\blabels?\s*:\s*`?(type|topic):",
    r"\b(list|mark|report) .{0,40} as missing\b",
]
INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

MAX_REASON_CHARS = 200


def redact(text: str) -> str:
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return HOME_PATHS.sub(lambda m: f"{m['pre']}<user>", text)


def fence(text: str) -> str:
    """Neutralise anything that looks like our fence tags, so text can't close
    the <issue> block early and continue as if it were instructions."""
    return re.sub(r"<(/?)\s*issue", r"<\1issue-text", text, flags=re.IGNORECASE)


def clean_for_prompt(text: str | None) -> str:
    return fence(redact(text or ""))


def injection_signals(text: str | None) -> list[str]:
    """Phrases that look like attempts to steer the model. Flagged, not blocked:
    blocking on regex would refuse legitimate issues ABOUT prompts or labels."""
    return sorted({m.group(0).lower() for m in INJECTION.finditer(text or "")})


def clean_reason(text: str) -> str:
    """Model-written reasons are shown to humans: cap length, drop links and pings."""
    text = re.sub(r"https?://\S+", "<link>", text)
    text = re.sub(r"@(\w)", r"@ \1", text)  # break @mentions so nothing pings anyone
    text = " ".join(text.split())
    return text[:MAX_REASON_CHARS]
