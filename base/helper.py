# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from datetime import datetime
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import json

from agents.base import Input, Output
from .logger import setup_logging
log = setup_logging()

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent:
    timestamp: datetime
    agent: str
    input: Input
    output: Output


class TracePersister:
    """Very small in‑memory trace store."""

    def __init__(self) -> None:
        self.events: List[TraceEvent] = []

    def save(self, event: TraceEvent) -> None:
        self.events.append(event)
        log.info("Trace saved: %s → %s", event.agent, str(event.output))

    def dump_json(self, n: int | None = None) -> str:
        subset = self.events if n is None else self.events[:n]
        return json.dumps([e.__dict__ for e in subset], default=str, indent=2)


class KBPersister:
    """Toy knowledge base; swap for a real DB later."""

    def __init__(self) -> None:
        self.rows: Dict[str, Dict[str, Any]] = {}

    def save(self, uid: str, data: Any, meta: Optional[Dict[str, Any]] = None) -> None:
        self.rows[uid] = {"data": data, "meta": meta or {}}
        log.info("KB save id=%s, meta=%s", uid, meta)

    def fetch(self, uid: str) -> Optional[Dict[str, Any]]:
        return self.rows.get(uid)