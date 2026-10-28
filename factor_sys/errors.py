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
    factor that raises something unrelated to the factor system, a panel that
    does not belong to the run it was asked for, or a quality checker that
    answers with something other than a report about the panel it was given.  A
    *quality* finding is never raised: it travels back inside
    :class:`~factor_sys.quality.FactorQualityReport`, so the caller decides what
    a partial cross-section is worth.  Errors raised by the layers underneath
    keep their own type -- a broken read stays a
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


class FactorStoreError(FactorError):
    """Raised when factor-panel storage or retrieval fails.

    The store speaks in one domain error for both directions: a ``write`` that
    cannot be honoured (a caller that is not storing a
    :class:`~factor_sys.contracts.FactorPanel`, an unusable root, a version that
    cannot name a directory, a failed temporary write or replace) and a ``read``
    that cannot be satisfied (an illegal key, no file for a legal key, a file
    that is not Parquet, metadata that is missing, unknown or contradicts the
    table, a table that no longer validates as a panel).  ``exists`` is the one
    deliberate exception: a legal key with no file is ``False``, not an error.

    Errors raised underneath keep their chain: a corrupt Parquet file or a
    panel that fails :class:`~factor_sys.contracts.FactorPanel` validation is
    reported with ``raise ... from exc``, so the original reason is never lost.
    """


class FactorRegistryError(FactorError):
    """Raised when a factor cannot be registered or looked up.

    A registry is a table of *names*, so everything that can go wrong with the
    table lands here: a name that is not a string or not snake_case, a name that
    is already taken, a name nobody registered, a factory that is not callable,
    one that builds something other than a :class:`~factor_sys.contracts.Factor`,
    or one that raises.

    A factory that raises a :class:`FactorError` of its own is **not** wrapped:
    the layer that knows what went wrong (an invalid ``FactorSpec``, for example)
    already says it precisely, and a caller may branch on that type.  Any other
    failure is reported here with the original exception as ``__cause__``, so the
    traceback still shows where it came from.
    """


class QuickAnalysisError(FactorError):
    """Raised when panels cannot be described as basic statistics.

    Only the *shape of the request* fails here, never the data it carries: a
    caller that hands the analyzer something which is not a
    :class:`~factor_sys.contracts.FactorPanel`, asks for a history from a single
    panel, mixes two factor names or versions, repeats an ``as_of_date``, or
    builds a :class:`~factor_sys.quick_analysis.FactorPanelStatistics` whose
    coverage, rate, deviation or finiteness contradicts its own counts.  A
    panel where nothing could be computed is **not** an error: it is described
    with ``valid_count == 0`` and ``None`` for the four moments, because "no
    usable number today" is a result, not a broken request.

    Attributes
    ----------
    problems:
        Every reason the request was refused, collected the way
        :class:`FactorContractError` collects them, so one failed call names all
        the offending panels at once instead of only the first.
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        super().__init__(_with_problems(message, self.problems))


class ForwardReturnError(FactorError):
    """Raised when a forward-return label cannot be built or cannot be trusted.

    Two different failures land here, and both are about the *label*, never
    about a factor:

    * a request that cannot be honoured -- something that is not a
      :class:`~factor_sys.contracts.FactorPanel`, a horizon that is not a
      positive number of sessions, a decision date that is not a session of the
      calendar, a reader that is not a
      :class:`~factor_sys.market_data.MarketDataReader`, a calendar name that
      cannot be resolved;
    * an answer that cannot be believed -- a reader that returns something other
      than a bar frame, a missing ``date``/``symbol``/``adjusted_close`` column,
      a duplicated ``(symbol, date)``, a symbol outside the universe, or a date
      outside ``[as_of_date, end_date]``.  A label built on such an answer would
      be silently wrong, so it is refused instead of interpreted.

    A *missing or unusable price* is deliberately **not** an error: that symbol's
    label travels back with ``is_available=False`` and ``forward_return=NaN``, so
    one gap never destroys the rest of the cross-section.

    Attributes
    ----------
    problems:
        Every reason the request or the label table was refused, collected the
        way :class:`FactorContractError` collects them, so one failed call names
        all the offending rows at once.
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        super().__init__(_with_problems(message, self.problems))


class ICAnalysisError(FactorError):
    """Raised when panels cannot be measured as one IC history.

    Only the *shape of the request* fails here -- never the data it carries:

    * an ``ICAnalyzer`` built with a ``min_pairs`` a correlation cannot use
      (anything below two, or not an integer at all);
    * two sequences that are not one factor's history: something that is not a
      panel, a single panel instead of an iterable, an empty sequence, two
      factor names or versions, two label horizons, or a repeated date;
    * a join that does not close: a date with a factor panel but no labels (or
      the other way round), an identity that differs between a panel and its
      labels, or a universe that moved or was re-ordered between them;
    * a hand-built :class:`~factor_sys.ic_analysis.ICObservation` or
      :class:`~factor_sys.ic_analysis.ICReport` whose fields contradict each
      other: a coverage its counts cannot produce, an IC outside ``[-1, 1]``, a
      count that disagrees with the observations, or a mean that is not the mean
      of the ICs it holds.

    An *undefined* correlation is deliberately **not** an error.  Too few pairs,
    or a sample that is constant on one side, is reported as ``ic=None`` and
    stays in the report, because "this session carries no IC" is a fact about
    the factor, not a broken request.

    Attributes
    ----------
    problems:
        Every reason the request or the report was refused, collected the way
        :class:`FactorContractError` collects them, so one failed call names all
        the offending panels at once.
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        super().__init__(_with_problems(message, self.problems))


class FactorScheduleError(FactorError):
    """Raised when a decision-date schedule cannot be selected as requested.

    Only the *rule* fails here, never the calendar it is evaluated against:

    * a ``frequency`` that is not ``weekly``, ``biweekly`` or ``monthly``;
    * an ``anchor_weekday`` that is not one of ``MON``..``FRI`` (or ``None``),
      including an anchor passed for a monthly cadence, which has no anchor;
    * a ``start_date``/``end_date`` that is not a plain date or a strict ISO
      ``"YYYY-MM-DD"`` string, or a window whose start is after its end;
    * a trading calendar that cannot be loaded;
    * a hand-built :class:`~factor_sys.factor_schedule.DecisionSchedule` whose
      fields contradict each other: an unknown cadence, an anchor a monthly
      schedule cannot have, dates outside the requested window, or dates that
      are not strictly ascending.

    An *empty* selection is deliberately **not** an error: a window may simply
    hold no decision day (a Thursday anchor over a two-day window that ends on
    Wednesday), and that is reported as an empty tuple of dates.
    """


class QuantileAnalysisError(FactorError):
    """Raised when panels cannot be measured as one quantile history.

    Only the *shape of the request* fails here -- never the data it carries:

    * a ``QuantileAnalyzer`` built with a ``quantiles`` that is not an integer of
      at least two (a ``bool``, a float, a string, one, zero or a negative);
    * two sequences that are not one factor's history: something that is not a
      panel, a single panel instead of an iterable, an empty sequence, two factor
      names or versions, two label horizons, or a repeated date;
    * a join that does not close: a date with a factor panel but no labels (or
      the other way round), an identity that differs between a panel and its
      labels, or a universe that moved or was re-ordered between them;
    * a hand-built :class:`~factor_sys.quantile_analysis.QuantileObservation`,
      :class:`~factor_sys.quantile_analysis.QuantileSpreadObservation` or
      :class:`~factor_sys.quantile_analysis.QuantileReport` whose fields
      contradict each other: a coverage its counts cannot produce, a mean on an
      empty group, a spread that is not the difference of its own extremes, a
      group count that disagrees with the observations, or a report whose member
      counts do not add up to the paired count of its session.

    An *empty group* or a *session with no usable symbol* is deliberately **not**
    an error.  It is reported with ``member_count=0``, ``mean_forward_return=None``
    and -- when an extreme group is empty -- ``top_minus_bottom=None``, because
    "this group has no member" is a fact about the cross-section, not a broken
    request.

    Attributes
    ----------
    problems:
        Every reason the request or the report was refused, collected the way
        :class:`FactorContractError` collects them, so one failed call names all
        the offending panels at once.
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        super().__init__(_with_problems(message, self.problems))


class FactorResearchError(FactorError):
    """Raised when one multi-date research analysis cannot be produced as asked.

    Only the *orchestration* of the research entry point fails here, never the
    data it measures:

    * a request that cannot be honoured -- a ``horizon_sessions`` that is not a
      positive whole number of sessions, a ``quantiles`` below two, a
      ``min_pairs`` a correlation cannot use, a blank or unusable ``data_root``
      or ``report_root``;
    * a selection with nothing to measure or with holes in it -- a schedule that
      selected no decision date, at least one selected date that failed (the
      analysis is refused instead of quietly measuring the dates that survived),
      or a batch whose manifest could not be written;
    * a report that cannot be published -- a directory that cannot be created,
      a table or a metadata file that cannot be written, or an incomplete or
      unpublishable staging directory, in which case nothing is left behind.

    Everything about *one date* stays with the layer that owns it: a damaged
    panel is a :class:`FactorStoreError` recorded by the batch, a missing price
    is an unavailable label (never an error), and a hard refusal of the label,
    IC or quantile layer keeps its own type -- a
    :class:`ForwardReturnError`, :class:`ICAnalysisError` or
    :class:`QuantileAnalysisError` travels back unchanged, so a caller can tell
    "the request was incomplete" from "the analysis refused these objects".

    Attributes
    ----------
    problems:
        Every reason the request, the selection or the report was refused,
        collected the way :class:`FactorContractError` collects them, so one
        failed call names all of them at once.
    """

    def __init__(self, message: str, problems: Sequence[str] | None = None) -> None:
        self.problems: list[str] = list(problems) if problems else []
        super().__init__(_with_problems(message, self.problems))


__all__ = [
    "FactorError",
    "FactorContractError",
    "MarketDataReadError",
    "FactorRunError",
    "FactorRegistryError",
    "FactorStoreError",
    "FactorScheduleError",
    "QuickAnalysisError",
    "ForwardReturnError",
    "ICAnalysisError",
    "QuantileAnalysisError",
    "FactorResearchError",
]
