"""评测指标 — 基于 jiwer 的 CER / Sub / Del / Ins / Hits 计算。"""

import re
from dataclasses import dataclass, field

import jiwer

# 中文标点 + 常见英文标点
_PUNCT_RE = re.compile(r"[，。！？、；：""''（）《》【】…—～·,.!?;:\"'()\[\]{}<>`\-]")


def _strip_punct(text: str) -> str:
    """去除空格和标点符号，返回纯文字字符串。"""
    return re.sub(r"\s+", "", _PUNCT_RE.sub("", text))


@dataclass
class ErrorMetrics:
    """一次识别结果的错误指标。"""

    cer: float = 0.0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0
    reference_length: int = 0  # 参考文本字数
    reference_text: str = ""
    hypothesis_text: str = ""


@dataclass
class AggregateMetrics:
    """聚合后的指标汇总。"""

    total_hits: int = 0
    total_substitutions: int = 0
    total_deletions: int = 0
    total_insertions: int = 0
    total_reference_length: int = 0
    total_utterances: int = 0
    cer: float = 0.0
    ser: float = 0.0  # 句错率（有错误的句子比例）
    sub_rate: float = 0.0
    del_rate: float = 0.0
    ins_rate: float = 0.0
    accuracy: float = 0.0  # 1 - CER


def compute_cer(
    reference: str,
    hypothesis: str,
    strip_punctuation: bool = True,
) -> ErrorMetrics:
    """计算单条识别结果的 CER 及其拆解。

    Args:
        reference: 标注文本
        hypothesis: 识别文本
        strip_punctuation: 是否去除标点符号后计算，默认 True

    Returns:
        ErrorMetrics 包含 CER 和 Sub/Del/Ins/Hits 拆解
    """
    if not reference and not hypothesis:
        return ErrorMetrics(reference_text=reference, hypothesis_text=hypothesis)

    if not reference:
        # 无参考但有输出 → 全部视为插入
        return ErrorMetrics(
            cer=1.0,
            insertions=len(hypothesis),
            reference_length=0,
            reference_text=reference,
            hypothesis_text=hypothesis,
        )

    if strip_punctuation:
        ref_clean = _strip_punct(reference)
        hyp_clean = _strip_punct(hypothesis)
    else:
        ref_clean = reference.replace(" ", "")
        hyp_clean = hypothesis.replace(" ", "")
    ref_len = len(ref_clean)

    output = jiwer.process_characters(ref_clean, hyp_clean)

    return ErrorMetrics(
        cer=output.cer,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        hits=output.hits,
        reference_length=ref_len,
        reference_text=reference,
        hypothesis_text=hypothesis,
    )


def aggregate_metrics(per_utterance: list[ErrorMetrics]) -> AggregateMetrics:
    """聚合多条识别结果，计算汇总指标。

    注意：聚合 CER 使用加权方式（总错误数 / 总参考字数），
    而非简单平均每条 CER。
    """
    if not per_utterance:
        return AggregateMetrics()

    total_hits = sum(m.hits for m in per_utterance)
    total_subs = sum(m.substitutions for m in per_utterance)
    total_dels = sum(m.deletions for m in per_utterance)
    total_inss = sum(m.insertions for m in per_utterance)
    total_ref_len = sum(m.reference_length for m in per_utterance)
    total_utt = len(per_utterance)

    total_errors = total_subs + total_dels + total_inss
    cer = total_errors / total_ref_len if total_ref_len > 0 else 0.0
    ser = sum(1 for m in per_utterance if m.cer > 0.01) / total_utt
    sub_rate = total_subs / total_ref_len if total_ref_len > 0 else 0.0
    del_rate = total_dels / total_ref_len if total_ref_len > 0 else 0.0
    ins_rate = total_inss / total_ref_len if total_ref_len > 0 else 0.0

    accuracy = round(total_hits / total_ref_len, 6) if total_ref_len > 0 else 0.0
    return AggregateMetrics(
        total_hits=total_hits,
        total_substitutions=total_subs,
        total_deletions=total_dels,
        total_insertions=total_inss,
        total_reference_length=total_ref_len,
        total_utterances=total_utt,
        cer=round(cer, 6),
        ser=round(ser, 4),
        sub_rate=round(sub_rate, 6),
        del_rate=round(del_rate, 6),
        ins_rate=round(ins_rate, 6),
        accuracy=accuracy,
    )
