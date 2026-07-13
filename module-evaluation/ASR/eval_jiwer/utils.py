"""系统信息与通用工具。"""

import logging
import platform
from datetime import datetime, timezone, timedelta


_CN_TZ = timezone(timedelta(hours=8))


def get_hardware_info() -> dict:
    """收集 CPU 与系统信息。"""
    import psutil

    cpu_info = {
        "cpu_model": platform.processor() or "unknown",
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 1),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
    }
    return cpu_info


def now_iso() -> str:
    """返回北京时间 ISO 8601 时间字符串。"""
    return datetime.now(_CN_TZ).isoformat(timespec="seconds")


def setup_logging(verbose: bool = False) -> None:
    """设置日志格式。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
