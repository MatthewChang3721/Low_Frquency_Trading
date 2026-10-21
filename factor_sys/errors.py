"""Exception hierarchy for the factor system.

The factor system is a **separate boundary** from the data pipeline: a factor
contract violation is not a :class:`~data_sys.errors.DataPipelineError`, so
``data_sys`` never has to know that factors exist, and a caller can catch the
whole factor system through :class:`FactorError`.
"""

from __future__ import annotations

from collections.abc import Sequence


class FactorError(Exception):
    """Base class for every error raised by the factor system."""


def _with_problems(message: str, problems: Sequence[str]) -> str:
    """Append collected problems to ``message`` in the project's bullet style."""
    if not problems:
        return message
    return f"{message}\n  - " + "\n  - ".join(problems)


class FactorContractError(FactorError):
    """Raised when factor input or output violates the shared contract.

    Attributes
    ----------
    problems:
        Human readable descriptions of every violation found while validating
        one object.  Validation of a frame deliberately *collects* instead of
        stopping at the first bad row (the style of
        :class:`data_sys.errors.UniverseError`), so one run reports every problem
        a frame has.
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        super().__init__(_with_problems(message, self.problems))


class MarketDataReadError(FactorError):
    """Raised when local market data cannot satisfy a read request.

    Both sides of a read can fail it: a *request* the system refuses to honour
    (an impossible date, a start after ``as_of_date``, an unknown or repeated
    field) and a *dataset* that cannot answer it (no root, no parquet file, a
    column the caller asked for and the data does not have).  Either way the
    caller gets one domain error naming what was wrong, never a bare
    ``KeyError``, ``FileNotFoundError`` or pyarrow message.

    Attributes
    ----------
    problems:
        Every reason the request was refused, collected the way
        :class:`FactorContractError` collects them, so one failed read reports
        all of them at once.
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        super().__init__(_with_problems(message, self.problems))


class FactorRunError(FactorError):
    """Raised when factor orchestration cannot produce a valid factor panel.

    Only *orchestration* failures are reported here: a dependency that does not
    implement its interface, a decision date that is not a trading session, a
    factor that raises something unrelated to the factor system, or a panel that
    does not belong to the run it was asked for.  Errors raised by the layers
    underneath keep their own type -- a broken read stays a
    :class:`MarketDataReadError`, a contract violation stays a
    :class:`FactorContractError` -- so a caller can tell "the data was not there"
    from "the formula was wrong" from "the run itself was impossible".

    Attributes
    ----------
    problems:
        Every reason the panel was refused, collected the way
        :class:`FactorContractError` collects them, so one failed run names all
        the ways its output mismatched at once.
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        super().__init__(_with_problems(message, self.problems))


__all__ = ["FactorError", "FactorContractError", "MarketDataReadError", "FactorRunError"]
