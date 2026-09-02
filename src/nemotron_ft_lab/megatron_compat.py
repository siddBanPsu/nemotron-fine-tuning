"""Narrow compatibility checks for the lab's pinned Megatron stack."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


EXPERT_BIAS_PADDING_MASK_MARKER = "_mbridge_expert_bias_padding_mask_compatible"


def _callable_source(function: Callable[..., Any]) -> str:
    """Return normalized source from a decorated callable and its common originals."""
    candidates = [function, inspect.unwrap(function)]
    dynamo_original = getattr(function, "_torchdynamo_orig_callable", None)
    if dynamo_original is not None:
        candidates.append(dynamo_original)

    sources: list[str] = []
    for candidate in candidates:
        try:
            source = inspect.getsource(candidate)
        except (OSError, TypeError):
            continue
        sources.append("".join(source.split()))
    return "\n".join(dict.fromkeys(sources))


def configure_expert_bias_padding_mask_compatibility(router_class: type) -> str:
    """Prevent Bridge from wrapping an MCore router that already expands the mask.

    Megatron-Bridge's packed-data compatibility wrapper expands a flat token mask
    to ``[tokens, 1]``. The MCore revision pinned by this lab already performs that
    expansion. Applying both fixes creates a three-dimensional routing map during
    broadcasting and breaks the expert counter update.

    Returns a short mode string for provenance logging. Older MCore revisions that
    still need the Bridge wrapper are left untouched.
    """
    method = router_class._apply_expert_bias
    if getattr(method, EXPERT_BIAS_PADDING_MASK_MARKER, False):
        return "already-compatible"

    source = _callable_source(method)
    native_expansion = "routing_map=routing_map&(~padding_mask).unsqueeze(-1)"
    bridge_needed = "routing_map=routing_map&(~padding_mask)"

    if native_expansion in source:
        setattr(method, EXPERT_BIAS_PADDING_MASK_MARKER, True)
        return "native-mcore"
    if bridge_needed in source:
        return "bridge-wrapper"

    raise RuntimeError(
        "Cannot determine the installed Megatron-Core expert-bias padding-mask behavior. "
        "Use the repository's pinned Megatron-Bridge checkout instead of an unverified stack."
    )
