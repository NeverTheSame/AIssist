"""logging.Filter that redacts detected PII from formatted log output.

Attach to individual handlers, not loggers: this scrubs what leaves the
process (files, stdout) while leaving in-memory LogRecord objects (and any
other handler you don't attach it to) alone. Catches the raw-prompt debug
block and summary dump in processor.py, the fetcher subprocess dump in
main.py, and any future leak, without editing individual log statements.
"""

import logging
from typing import Iterable, List

from .detectors import Detector, merge_spans
from .vault import PseudonymVault


class RedactionLogFilter(logging.Filter):
    def __init__(self, detectors: Iterable[Detector], vault: PseudonymVault):
        super().__init__()
        self._detectors = list(detectors)
        self._vault = vault

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True

        spans = []
        for detector in self._detectors:
            try:
                spans.extend(detector.detect(message))
            except Exception:
                logger = logging.getLogger(__name__)
                logger.debug("Log redaction detector %s failed; leaving message as-is for this detector", detector.name)
                continue

        if spans:
            spans = merge_spans(spans)
            record.msg = self._vault.redact(message, spans)
            record.args = ()

        return True


def install(handlers: Iterable[logging.Handler], detectors: List[Detector], vault: PseudonymVault) -> None:
    redaction_filter = RedactionLogFilter(detectors, vault)
    for handler in handlers:
        handler.addFilter(redaction_filter)
