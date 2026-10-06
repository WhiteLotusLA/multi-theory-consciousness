"""Command-line interface for the framework (the mtc console script).

``mtc.cli.main`` holds the parser and the entry point, ``mtc.cli.doctor`` the
install checks, ``mtc.cli.http`` the small client behind ``mtc chat`` and
``mtc ctm``, and ``mtc.cli.assess`` the assessment runner. ``main`` is loaded
on first use, so importing the package stays cheap.
"""

from typing import Any, List

__all__: List[str] = ["main"]


def __getattr__(name: str) -> Any:
    if name == "main":
        from mtc.cli.main import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
