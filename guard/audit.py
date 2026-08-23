"""JSONL audit trail for every guarded LLM call.

One record per call, written to GUARD_AUDIT_DIR (gitignored, size-rotated).
Field names are chosen to map onto a Log Analytics custom table -- this is
an audit trail with a schema designed for evidence collection, not SOC 2
evidence on its own (see docs/THREAT_MODEL.md).
"""

import dataclasses
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_BYTES = 10 * 1024 * 1024
_lock = threading.Lock()


@dataclasses.dataclass
class AuditRecord:
    call_id: str
    incident_id: Optional[str]
    call_site: Optional[str]
    model: Optional[str]
    entity_counts: Dict[str, int]
    tiers_fired: List[str]
    latency_ms: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    injection_verdict: Optional[str]
    rehydration_mismatches: List[str]
    image_count: int
    fail_open_triggered: bool
    estimated_cost: Optional[float] = None
    timestamp: str = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), default=str)


def new_call_id() -> str:
    return uuid.uuid4().hex


def _rotate_if_needed(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size > _MAX_BYTES:
            rotated = path.with_suffix(path.suffix + f".{int(time.time())}")
            path.rename(rotated)
    except OSError:
        logger.exception("Failed to rotate audit log %s", path)


def write_audit_record(audit_dir: str, record: AuditRecord) -> None:
    directory = Path(audit_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "audit.jsonl"
        with _lock:
            _rotate_if_needed(path)
            with open(path, "a", encoding="utf-8") as f:
                f.write(record.to_json() + "\n")
    except OSError:
        logger.exception("Failed to write guard audit record")
