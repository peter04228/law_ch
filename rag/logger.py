# rag/logger.py
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

def append_jsonl(
    path: str,
    record: Dict[str, Any],
    ensure_ascii: bool = False,
) -> None:
    """
    Append one record as a JSON line.
    - Creates parent directories automatically.
    - Uses UTF-8.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 타임스탬프 기본 추가 (없으면)
    record.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    line = json.dumps(record, ensure_ascii=ensure_ascii)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")