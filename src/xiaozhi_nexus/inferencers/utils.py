from __future__ import annotations


def format_text_for_log(text: str, max_len: int = 20) -> tuple[str, int]:
    text = text or ""
    text_len = len(text)
    if text_len > max_len:
        return f"{text[:max_len]}...", text_len
    return text, text_len


def audio_duration_seconds(sample_count: int, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return sample_count / float(sample_rate)
