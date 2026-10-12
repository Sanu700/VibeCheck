"""
Linguistic and Semantic Feature Utilities for ToxicBuddy / VibeCheck.
Provides text enrichment for sarcasm, sentiment incongruity, vowel elongation,
quoted irony markers, and conversational context cues.
"""

import re
from typing import List, Optional
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

POSITIVE_PRAISE_WORDS = {
    "lovely", "invaluable", "masterpiece", "incredible", "brilliant", "genius", "wisdom",
    "art", "hardest", "flawless", "amazing", "wonderful", "grace", "perfection", "expert",
    "mastermind", "rockstar", "great", "best", "superb", "phenomenal", "stellar"
}

NEGATIVE_FAILURE_WORDS = {
    "ruin", "trainwreck", "disaster", "sadder", "wrong", "useless", "messed", "zero",
    "waste", "fail", "failure", "break", "broke", "broken", "catastrophe", "incompetence",
    "late", "mess", "garbage", "crap", "terrible", "horrible", "trash"
}

FAILURE_CONTEXT_KEYWORDS = {
    "crashed", "broke", "broken", "failed", "failure", "bug", "error", "mistake",
    "deleted", "dropped", "outage", "down", "delay", "ruined", "disaster", "crash"
}

CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS - {
    "not", "no", "never", "neither", "nor", "none", "like", "last", "least", "less", "more", "most"
})

def enrich_text_features(text: str, context: Optional[List[str]] = None) -> str:
    """
    Enriches text with semantic and structural tokens to capture subtle sarcasm,
    sentiment contrast, elongation, and conversational context.
    """
    if not text or not text.strip():
        return ""

    augmented_text = text.strip()
    text_lower = augmented_text.lower()
    tokens = set(re.findall(r"\b[a-z]+\b", text_lower))
    tags = []

    # 1. Elongation detection & base-word normalization (e.g., 'suuuure' -> 'sure')
    if re.search(r"([a-zA-Z])\1{2,}", augmented_text):
        normalized = re.sub(r"([a-zA-Z])\1{2,}", r"\1", augmented_text)
        augmented_text = f"{augmented_text} {normalized} __elongated__"

    # 2. Quoted praise markers (e.g. "genius", 'expert')
    quoted_words = re.findall(r"[\"\'\“\‘]([a-zA-Z]+)[\"\'\”\’]", text_lower)
    if any(w in POSITIVE_PRAISE_WORDS for w in quoted_words):
        tags.append("__quoted_praise__")

    # 3. Sentiment contrast / valence incongruity
    has_pos = bool(tokens.intersection(POSITIVE_PRAISE_WORDS))
    has_neg = bool(tokens.intersection(NEGATIVE_FAILURE_WORDS))
    if has_pos and has_neg:
        tags.append("__sentiment_contrast__")

    # 4. Contextual failure irony
    if context:
        recent_context = " ".join(context[-2:]).lower()
        context_has_failure = any(w in recent_context for w in FAILURE_CONTEXT_KEYWORDS)
        if context_has_failure and has_pos and len(tokens) <= 12:
            tags.append("__context_sarcastic_praise__")

    if tags:
        return f"{augmented_text} {' '.join(tags)}"
    return augmented_text
