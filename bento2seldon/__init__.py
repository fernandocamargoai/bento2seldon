# type: ignore[attr-defined]
"""This project aims to combine the awesome capabilities of BentoML in packaging models with the powerful Seldon Core engine to deploy such models. It also features an optional cache using Redis that can also be used to make the feedback loop easier by using the request ID to get back the original request and response. For now, it was created for internal use and is in alpha state. But it will soon be prepared to be used by everyone."""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
