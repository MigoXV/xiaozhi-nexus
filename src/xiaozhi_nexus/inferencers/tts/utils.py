"""
TTS 文本预处理工具函数
"""

from __future__ import annotations

import re
from typing import List


def clean_text_for_tts(text: str) -> str:
    """
    清理文本以供 TTS 使用

    - 去除换行符
    - 去除多余空白

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    # 去掉换行符
    text = text.replace("\n", "").replace("\r", "")
    return text


def split_text_by_punctuation(text: str) -> List[str]:
    """
    按标点符号切分文本为多个短句

    支持中英文标点符号，包括：
    - 句号（。.）
    - 问号（？?）
    - 感叹号（！!）
    - 分号（；;）
    - 逗号（，,）- 可选，用于更细粒度的切分
    - 省略号（...、……）等连续标点

    Args:
        text: 要切分的文本

    Returns:
        切分后的句子列表
    """
    if not text:
        return []

    # 使用正则表达式按标点切分，保留标点符号
    # 匹配连续的中英文标点符号（如省略号 ...、……、？！等）
    pattern = r'([。？！；.?!;]+)'
    parts = re.split(pattern, text)

    # 将标点符号合并到前面的文本中
    sentences = []
    current = ""
    for part in parts:
        if re.match(pattern, part):
            # 这是标点符号（可能是连续的），合并到当前句子
            current += part
            if current.strip():
                sentences.append(current)
            current = ""
        else:
            current = part

    # 处理最后可能没有标点的文本
    if current.strip():
        sentences.append(current)

    return sentences
