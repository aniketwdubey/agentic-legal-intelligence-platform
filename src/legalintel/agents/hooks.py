"""Cross-cutting agent hooks.

Strands ``hooks`` are the idiomatic place for behaviour that wraps every agent
invocation — logging, timing, monitoring, guardrails — instead of scattering log
calls through each agent's method (or a bespoke logger-wrapper). ``LoggingHook``
records invocation start/stop; attach it to any agent via ``hooks=[LoggingHook()]``.

Note: citation-verification is deliberately NOT a hook — it is a first-class,
deterministic pipeline stage (see ``validation/validator.py``). Hooks here are for
observability, not for the grounding guarantee.
"""

from __future__ import annotations

import structlog
from strands.hooks import AfterInvocationEvent, BeforeInvocationEvent, HookProvider, HookRegistry

log = structlog.get_logger("agent")


class LoggingHook(HookProvider):
    """Emit a structured log line at the start and end of each agent invocation."""

    def register_hooks(self, registry: HookRegistry, **_: object) -> None:
        registry.add_callback(BeforeInvocationEvent, self._before)
        registry.add_callback(AfterInvocationEvent, self._after)

    def _before(self, event: BeforeInvocationEvent) -> None:
        log.debug("agent.start", agent=getattr(event.agent, "name", None))

    def _after(self, event: AfterInvocationEvent) -> None:
        log.debug("agent.end", agent=getattr(event.agent, "name", None))
