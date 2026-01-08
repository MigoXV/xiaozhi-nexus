import re
from collections.abc import Iterable

EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"  # supplemental symbols
    "\U0001fa00-\U0001faff"
    "\u2600-\u26ff"  # misc symbols
    "\u2700-\u27bf"  # dingbats
    "]+",
    flags=re.UNICODE,
)


def remove_emoji(text: str) -> str:
    return EMOJI_PATTERN.sub("", text)


def chunk_text_normalization(text: str) -> str:
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("*", "").replace("_", "").replace("~", "")
    text = remove_emoji(text)  # 去除emoji
    return text


def stream_text_normalization(text_itr: Iterable[str]) -> Iterable[str]:
    text_itr = (chunk_text_normalization(s) for s in text_itr if s.strip())
    return text_itr

def normalize_full_text(text: str) -> str:
    """对完整文本进行归一化处理"""
    text = chunk_text_normalization(text)
    return text
