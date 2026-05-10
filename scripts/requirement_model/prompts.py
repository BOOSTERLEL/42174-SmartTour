"""Prompt templates for English requirement data augmentation."""

from __future__ import annotations

import json
import re

MARKER_PATTERN = re.compile(r"\[[A-Z_]+\|[^\[\]\|]+\]")

SYSTEM_PROMPT = (
    "You generate English travel-planning training utterances for a slot-filling "
    "model. Treat every inline slot marker as immutable text. Preserve each "
    "marker exactly as provided, including the slot label and value. Do not "
    "invent marker values, translate marker values, change [LANGUAGE|English], "
    "or add unsupported markers. Return only a JSON object with an utterances "
    "array."
)


def build_augmentation_messages(
    marked_examples: list[str], target_count: int
) -> list[dict[str, str]]:
    """
    Build chat messages for marker-preserving LLM augmentation.

    Args:
        marked_examples: Seed utterances containing inline slot markers.
        target_count: The requested number of augmented utterances.

    Returns:
        Chat messages suitable for the OpenAI SDK chat completions interface.
    """
    allowed_markers = sorted(
        {marker for example in marked_examples for marker in extract_markers(example)}
    )
    payload = {
        "target_count": target_count,
        "requirements": [
            "Write natural English travel-planning requests.",
            "Rewrite only the unmarked wording around markers.",
            "Copy every marker from the allowed_markers list exactly.",
            "Do not create any marker that is not in allowed_markers.",
            "Keep each utterance realistic and different from the seeds.",
            'Return JSON only: {"utterances": ["..."]}.',
        ],
        "allowed_markers": allowed_markers,
        "marked_examples": marked_examples,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=True, indent=2),
        },
    ]


def extract_markers(marked_text: str) -> list[str]:
    """
    Return inline slot markers from marked text.

    Args:
        marked_text: The source marked text.

    Returns:
        The discovered markers.
    """
    return MARKER_PATTERN.findall(marked_text)
