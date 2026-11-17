class CybORGAdapter:
    """Fail-closed boundary: validate against a pinned CybORG release before use."""
    def __init__(self, *args, **kwargs):
        try:
            import CybORG  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install CybORG from https://github.com/cage-challenge/CybORG") from exc
        raise NotImplementedError(
            "CybORG import detected, but API/action/observation mapping is not validated "
            "for this installed commit. Inspect and pin the selected release first."
        )
