"""Per-call incident context, threaded through to audit records.

A contextvar rather than a constructor argument because the enforcement
point (azure_auth.get_openai_client_with_auth) builds the client long
before any particular incident is known; callers set the current incident
once per processing loop and every guarded call made within it picks the
id up automatically.
"""

import contextvars
from typing import Optional

_incident_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "guard_incident_id", default=None
)
_call_site: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "guard_call_site", default=None
)


def set_incident_context(incident_id: Optional[str], call_site: Optional[str] = None) -> None:
    _incident_id.set(incident_id)
    if call_site is not None:
        _call_site.set(call_site)


def get_incident_id() -> Optional[str]:
    return _incident_id.get()


def get_call_site() -> Optional[str]:
    return _call_site.get()
