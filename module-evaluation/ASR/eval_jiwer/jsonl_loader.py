"""JSONL 格式数据集加载器。

Manifest 格式（每行一条）:
    {"audio_path": "相对路径.wav", "text": "标注文本", "duration": 1.234}

audio_path 相对于 manifest 文件所在目录解析。
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Utterance:
    """单条话语记录。"""
    audio_path: str           # 解析后的绝对路径
    reference_text: str       # 标注文本
    dataset_name: str         # 数据集名称
    metadata: dict[str, Any]  # 额外元数据（duration 等）


class JsonlLoader:
    """JSONL 数据集加载器。

    属性:
        manifest_path: JSONL 文件路径
        max_hours: 最大加载时长（小时），0 表示不限制
        max_utterances: 最大加载条数，0 表示不限制
    """

    def __init__(
        self,
        manifest_path: str | Path,
        max_hours: float | None = None,
        max_utterances: int | None = None,
    ) -> None:
        self._manifest_path = Path(manifest_path).resolve()
        self._max_hours = max_hours or 0
        self._max_utterances = max_utterances or 0

    def name(self) -> str:
        """数据集名称，取 manifest 文件 stem。"""
        return self._manifest_path.stem

    def load(self) -> list[Utterance]:
        """加载并返回 Utterance 列表，受 max_hours / max_utterances 限制。"""
        if not self._manifest_path.exists():
            logger.error(f"Manifest not found: {self._manifest_path}")
            return []

        manifest_dir = self._manifest_path.parent
        utterances: list[Utterance] = []
        total_duration_s: float = 0.0
        skipped_invalid = 0

        with open(self._manifest_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # 检查条数限制
                if self._max_utterances > 0 and len(utterances) >= self._max_utterances:
                    break

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"[{self._manifest_path}:{line_no}] Invalid JSON: {e}"
                    )
                    continue

                audio_rel = record.get("audio_path", "")
                text = record.get("text", "")
                duration = float(record.get("duration", 0))

                if not audio_rel or not text:
                    skipped_invalid += 1
                    continue

                # 解析为绝对路径
                audio_abs = str((manifest_dir / audio_rel).resolve())

                # 检查时长限制
                if self._max_hours > 0:
                    if total_duration_s + duration > self._max_hours * 3600:
                        break
                    total_duration_s += duration

                utterances.append(
                    Utterance(
                        audio_path=audio_abs,
                        reference_text=text,
                        dataset_name=self.name(),
                        metadata={"duration": duration, "line_no": line_no},
                    )
                )

        if skipped_invalid:
            logger.warning(
                f"Skipped {skipped_invalid} records with missing audio_path/text"
            )

        logger.info(
            f"Loaded {len(utterances)} utterances "
            f"from {self._manifest_path.name} "
            f"(total audio: {total_duration_s:.1f}s / {total_duration_s / 3600:.2f}h)"
        )
        return utterances
