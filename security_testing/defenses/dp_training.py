"""
Optional differential privacy integrations.

Opacus-compatible hooks are fragile with 4-bit PEFT setups. For audited DP training prefer
purpose-built stacks (PrivLib, dp-transformers) and reconcile accounting with compliance teams.

Expose simple helpers so experimentation can incrementally migrate without coupling to train.py yet.
"""


def noop_attach(**_kwargs) -> None:  # pragma: no cover - placeholder shim
    return None


try:  # pragma: no cover
    import opacus  # noqa: F401

    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False


__all__ = ["noop_attach", "OPACUS_AVAILABLE"]
