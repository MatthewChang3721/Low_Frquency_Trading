"""Run one strategy for one decision date: arguments -> registry -> runner -> selection.

What this script is
-------------------
The minimal, real entry point of the strategy layer.  It owns no scoring rule, no
storage rule and no factor formula: it joins the modules that already exist, in
exactly this order::

     1  FactorRequirement(name, version)   the exact factor identity to read
     2  TopNByFactorStrategy(...)          the rule, built by the factory below
     3  StrategyRegistry()                 a table for this invocation only
     4  registry.register(factory)         one factory, one complete StrategySpec
     5  registry.create(spec)              the instance this run really uses
     6  StaticUniverseProvider(path)       the static stock pool
     7  ParquetFactorStore(panels root)    the panels the factor system stored
     8  StrategyRunner(provider, store)    the two interfaces, wired together
     9  runner.run(strategy, as_of_date)   one validated, provenance-checked result
    10  print the selection                stable ``key: value`` lines and a table

Usage (project root, virtual environment active)::

    python scripts/run_strategy.py --factor momentum_12_1 `
        --factor-version 1.0.0 --top-n 10 --as-of 2024-12-31

    python scripts/run_strategy.py --factor momentum_12_1 `
        --factor-version 1.0.0 --top-n 10 --as-of 2024-12-31 `
        --data-root C:\\research\\data --universe-path C:\\research\\pool.csv

Reading a selection, not making one
-----------------------------------
The command *reads* factor panels that the factor system already stored, under
``<data-root>/factors/panels``; it never computes a factor, never repairs a panel
and never writes one.  On a machine without the panel for the requested date it
fails with a readable error instead of recomputing, downloading or moving the date
to a neighbouring session.  ``--factor-version`` is required rather than defaulted
because a selection is only traceable if the exact factor version it was scored
from travels with it: the same factor name at another version is another panel and
another strategy identity.

The panel is read by one rule and one rule only: the ``(factor_name,
factor_version)`` pair on the command line becomes the strategy's
:class:`~strategy_sys.contracts.FactorRequirement`, so the store lookup is keyed by
exactly what was asked for -- there is no fallback to another version.

What the registry is doing here
-------------------------------
``StrategyRegistry`` has no default instance and no auto-registration, and this
script does not add one: it builds a table for this one command, registers a
factory that rebuilds the same rule, and runs whatever ``create`` hands back.  The
strategy that runs is therefore a registry-built instance, never the literal
argument, and the spec it was created under is the spec the runner checks the
result against.

The output
----------
One stable ``key: value`` block, then the selection::

    strategy_name: top_10_by_momentum_12_1
    strategy_version: 1.0.0
    as_of_date: 2024-12-31
    universe_size: 12
    selected_count: 3
    required_factor: momentum_12_1/1.0.0
    selection:
    rank  symbol  score
    1     NVDA    0.950000
    2     MSFT    0.800000
    3     META    0.700000

Only rows flagged ``is_selected`` are listed, in rank order, with six decimals; the
result itself keeps one row per universe symbol in universe order, untouched.  An
empty selection is a fact about the cross-section, not a failure: the command still
exits ``0``, prints the header and the line ``none``.

Exit codes
----------
``0``
    The strategy ran and its selection was printed, even when nothing was selected.
``1``
    A domain or orchestration error: an invalid strategy parameter
    (:class:`~strategy_sys.errors.StrategyContractError`), a missing or damaged panel
    (:class:`~factor_sys.errors.FactorStoreError`), an unreadable universe
    (:class:`~factor_sys.errors.FactorContractError`), a date the runner refuses
    (:class:`~strategy_sys.errors.StrategyRunError`), a registry failure
    (:class:`~strategy_sys.errors.StrategyRegistryError`) or a blank path
    (:class:`RunStrategyError`).  The message goes to stderr as one
    ``error: ...`` line, never as a traceback, and nothing is written.
``2``
    argparse's own code for a missing, malformed or unknown flag.

This is not a backtest
----------------------
A selection is a decision-date list of symbols.  There are no weights, no cash, no
positions, no orders, no fills, no costs and no returns here; portfolio
construction, execution and simulation come later, on top of this result.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import NamedTuple

# Running this file directly puts ``scripts/`` (not the project root) on
# ``sys.path``; add the root so ``python scripts/run_strategy.py`` works from a
# checkout, with or without the package installed in the environment.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_sys.config import PipelineConfig  # noqa: E402
from factor_sys import (  # noqa: E402
    FactorError,
    ParquetFactorStore,
    StaticUniverseProvider,
)
from strategy_sys import (  # noqa: E402
    FactorRequirement,
    StrategyError,
    StrategyRegistry,
    StrategyResult,
    StrategyRunError,
    StrategyRunner,
    StrategySpec,
)
from strategy_sys.strategies import TopNByFactorStrategy  # noqa: E402

#: Strategy version recorded when ``--strategy-version`` is not given.
DEFAULT_STRATEGY_VERSION = "1.0.0"

#: Decimals a score is printed with.
SCORE_DECIMALS = 6

#: Width of the ``rank`` column of the selection table.
RANK_WIDTH = 6

#: Width of the ``symbol`` column of the selection table.
SYMBOL_WIDTH = 8

#: Printed under the table header when nothing was selected.
EMPTY_SELECTION_TEXT = "none"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class RunStrategyError(StrategyRunError):
    """A request this entry point refuses before any module could answer it.

    Only *argument* problems land here -- a blank ``--data-root``, a path of the
    wrong type, an outcome whose parts do not describe one run -- because everything
    else stays with the layer that owns the rule: a bad factor identity or a bad
    ``top_n`` is a
    :class:`~strategy_sys.errors.StrategyContractError`, a missing panel is a
    :class:`~factor_sys.errors.FactorStoreError`, an unreadable pool is a
    :class:`~factor_sys.errors.FactorContractError` and a refused date is a
    :class:`~strategy_sys.errors.StrategyRunError`.  All of them share
    :class:`~strategy_sys.errors.StrategyError` or
    :class:`~factor_sys.errors.FactorError`, so the command line catches them the
    same way and never has to guess.

    It derives from :class:`~strategy_sys.errors.StrategyRunError` only to reuse that
    error's collected-problem style (``problems`` plus the project's bullet
    rendering); every rule behind a refusal is still decided by the layer that owns
    it, never restated here.
    """


# ---------------------------------------------------------------------------
# What one command selected
# ---------------------------------------------------------------------------
class Selection(NamedTuple):
    """One selected symbol: its rank, its symbol and its score."""

    rank: int
    symbol: str
    score: float


@dataclass(frozen=True)
class StrategyRunOutcome:
    """Everything one :func:`run_strategy` call produced -- and nothing else.

    The object carries the run itself, not a rendering of it: the
    :class:`~strategy_sys.contracts.StrategyResult` the runner accepted, how many of
    its rows are selected, the exact factor identities it was scored from and the
    data root the panels were read from.  :func:`format_summary` is the only thing
    that turns it into text, so a test can assert on numbers instead of parsing
    stdout, and a notebook can call :func:`run_strategy` directly.

    No frame is copied into the outcome: the selected rows are derived from
    :attr:`result` on demand (:attr:`selections`), so the object stays small and a
    mutation of a frame handed out by ``result.frame`` can never make the outcome
    disagree with itself.

    Parameters
    ----------
    result:
        The one cross-section this run produced: one row per universe symbol, in
        universe order, with the strategy's complete spec attached.
    selected_count:
        How many rows of ``result`` are flagged ``is_selected``.  It must agree with
        the frame (and therefore with ``result.selected_symbols``); that is checked
        here, so a printed count can never contradict the printed table.
    factor_requirements:
        The exact ``(factor_name, factor_version)`` pairs the result was scored
        from -- equal to ``result.factor_requirements``, never re-derived here.
    data_root:
        The root the panels were read from (``<data_root>/factors/panels``), so a
        printed selection can be traced back to one data tree.

    Raises
    ------
    RunStrategyError
        If the parts do not describe one run: a result that is not a
        :class:`~strategy_sys.contracts.StrategyResult`, a ``selected_count`` that
        disagrees with the frame, factor requirements that are not the result's own,
        or a ``data_root`` that is not a path.
    """

    result: StrategyResult
    selected_count: int
    factor_requirements: tuple[FactorRequirement, ...]
    data_root: Path

    def __post_init__(self) -> None:
        """Refuse an outcome whose parts do not belong to the same run."""
        if not isinstance(self.result, StrategyResult):
            raise RunStrategyError(
                "result must be a StrategyResult, got "
                f"{type(self.result).__name__}: a frame, a tuple or a dictionary cannot say "
                "which strategy version or factor version it came from"
            )
        if not isinstance(self.data_root, Path):
            raise RunStrategyError(
                f"data_root must be a Path, got {type(self.data_root).__name__}: a string "
                "would leave the panels root ambiguous"
            )

        problems: list[str] = []
        requirements = self.factor_requirements
        if not isinstance(requirements, tuple):
            problems.append(
                f"factor_requirements must be a tuple, got {type(requirements).__name__}"
            )
        elif not all(isinstance(item, FactorRequirement) for item in requirements):
            problems.append(
                "factor_requirements must hold FactorRequirement values, got "
                f"{[type(item).__name__ for item in requirements]}"
            )
        elif requirements != self.result.factor_requirements:
            problems.append(
                f"factor_requirements are {_describe_requirements(requirements)}, but the "
                "result was scored from "
                f"{_describe_requirements(self.result.factor_requirements)}"
            )

        if isinstance(self.selected_count, bool) or not isinstance(self.selected_count, int):
            problems.append(
                "selected_count must be an int, got "
                f"{type(self.selected_count).__name__}: {self.selected_count!r}"
            )
        else:
            flagged = int(self.result.frame["is_selected"].astype(bool).sum())
            if self.selected_count != flagged:
                problems.append(
                    f"selected_count is {self.selected_count}, but {flagged} row(s) of the "
                    "result are flagged is_selected"
                )

        if problems:
            raise RunStrategyError("invalid StrategyRunOutcome", problems)

    @property
    def strategy_spec(self) -> StrategySpec:
        """The complete identity of the strategy that produced the result."""
        return self.result.strategy_spec

    @property
    def strategy_name(self) -> str:
        """``top_10_by_momentum_12_1`` -- the name the rule gives itself."""
        return self.result.strategy_spec.name

    @property
    def strategy_version(self) -> str:
        """The strategy version, taken from that same spec."""
        return self.result.strategy_spec.version

    @property
    def as_of_date(self) -> date:
        """The decision date this selection belongs to."""
        return self.result.as_of_date

    @property
    def universe_size(self) -> int:
        """How many symbols the pool held: the result has one row per symbol."""
        return len(self.result)

    @property
    def selected_symbols(self) -> tuple[str, ...]:
        """The selected symbols in the result's own (universe) row order."""
        return self.result.selected_symbols

    @property
    def selections(self) -> tuple[Selection, ...]:
        """The selected rows by rank ascending -- the printed table, as data."""
        frame = self.result.frame
        selected = frame.loc[frame["is_selected"].astype(bool), ["rank", "symbol", "score"]]
        ordered = selected.sort_values("rank")
        return tuple(
            Selection(rank=int(row.rank), symbol=str(row.symbol), score=float(row.score))
            for row in ordered.itertuples(index=False)
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(strategy_name={self.strategy_name!r}, "
            f"strategy_version={self.strategy_version!r}, "
            f"as_of_date={self.as_of_date.isoformat()!r}, "
            f"selected_count={self.selected_count!r})"
        )


def _describe_requirements(requirements: tuple[FactorRequirement, ...]) -> str:
    """``[momentum_12_1/1.0.0]`` -- the exact identities, in declared order."""
    return "[" + ", ".join(str(requirement) for requirement in requirements) + "]"


# ---------------------------------------------------------------------------
# The composition
# ---------------------------------------------------------------------------
def run_strategy(
    *,
    factor_name: str,
    factor_version: str,
    top_n: int,
    as_of_date: date | str,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
    data_root: Path | str | None = None,
    universe_path: Path | str | None = None,
) -> StrategyRunOutcome:
    """Run one Top N strategy for one session from panels that are already stored.

    The steps of the module docstring run strictly in order, each through the public
    interface of the module that owns it: no formula is imported here, no
    :class:`~strategy_sys.contracts.StrategyContext` is built by hand, no Parquet file
    is opened here, no scoring rule lives in this file and the store's layout is never
    reproduced here.

    Nothing is written: the panels are read, the strategy scores the cross-section and
    the result is returned.  If the requested date has no stored panel, the store's
    :class:`~factor_sys.errors.FactorStoreError` is raised unchanged -- the date is
    never moved to a neighbouring session, no factor is recomputed and no vendor is
    contacted.

    Parameters
    ----------
    factor_name:
        Factor the rule ranks by, e.g. ``"momentum_12_1"``.  It is validated by
        :class:`~strategy_sys.contracts.FactorRequirement`, which also refuses a blank
        name, so a typo fails before any file is opened.
    factor_version:
        The *exact* version of that factor to read, e.g. ``"1.0.0"``.  It is part of
        the strategy's identity, so a result can always say which panel it came from.
    top_n:
        How many rankable symbols to select: a positive integer.  The rule's own
        constructor validates it, refusing a bool, a float or a string instead of
        converting them.
    as_of_date:
        The decision date, as a :class:`datetime.date` or a strict ISO-8601
        ``"YYYY-MM-DD"`` string.  It is handed to the runner unchanged: a timestamp, a
        malformed string or a date without a stored panel is refused rather than
        adjusted.
    strategy_version:
        The strategy version recorded in the spec and in every row of the result.
    data_root:
        Root the panels are read from: ``<data_root>/factors/panels``.  ``None`` means
        :attr:`data_sys.config.PipelineConfig.data_root`.
    universe_path:
        The static stock-pool CSV.  ``None`` means the committed
        ``config/universe_seed.csv``.

    Returns
    -------
    StrategyRunOutcome
        The result the runner accepted, its selected count, the exact factor
        requirements it was scored from and the data root used.

    Raises
    ------
    strategy_sys.errors.StrategyError
        An invalid strategy parameter
        (:class:`~strategy_sys.errors.StrategyContractError`), a real orchestration
        failure (:class:`~strategy_sys.errors.StrategyRunError`), a registry failure
        (:class:`~strategy_sys.errors.StrategyRegistryError`) or a refused lookup.
    factor_sys.errors.FactorError
        A missing or damaged panel
        (:class:`~factor_sys.errors.FactorStoreError`) or an unreadable universe
        (:class:`~factor_sys.errors.FactorContractError`).
    RunStrategyError
        If ``data_root`` is blank or of a type no path can be made from.
    """
    config = _pipeline_config(data_root)

    # (1)-(2): the exact identity to read, and the rule that reads it.  A closure
    # rather than a value: the registry insists on building a new strategy per call,
    # and this is the only place that knows the rule's parameters.
    requirement = FactorRequirement(factor_name, factor_version)

    def build_strategy() -> TopNByFactorStrategy:
        """Build the requested rule again -- same factor, same size, same version."""
        return TopNByFactorStrategy(requirement, top_n, version=strategy_version)

    # (3)-(5): a table for this invocation only.  The registration probe validates the
    # parameters, and ``create`` hands back the instance that actually runs -- the
    # probe itself never reaches the runner.
    registry = StrategyRegistry()
    registered_spec = registry.register(build_strategy)
    strategy = registry.create(registered_spec)

    # (6)-(8): the replaceable parts, each through its own public constructor.  The
    # runner only knows the two interfaces, so the static pool can be replaced later
    # without this file changing.
    universe_provider = StaticUniverseProvider(universe_path)
    factor_store = ParquetFactorStore(_panels_root(config))
    runner = StrategyRunner(universe_provider, factor_store)

    # (9): one validated cross-section for exactly the date that was asked for.
    result = runner.run(strategy, as_of_date)

    # (10) happens in ``main``: this function answers with data, never with text.
    return StrategyRunOutcome(
        result=result,
        selected_count=int(result.frame["is_selected"].astype(bool).sum()),
        factor_requirements=result.factor_requirements,
        data_root=config.data_root,
    )


def _pipeline_config(data_root: Path | str | None) -> PipelineConfig:
    """The layout every path of this run is derived from."""
    if data_root is None:
        return PipelineConfig()
    if isinstance(data_root, str) and not data_root.strip():
        raise RunStrategyError(
            "--data-root must not be blank: pass a directory, or omit it to read the "
            f"project's default {PipelineConfig().data_root}"
        )
    if not isinstance(data_root, (str, Path)):
        raise RunStrategyError(
            f"data_root must be a path or a string, got {type(data_root).__name__}"
        )
    return PipelineConfig(data_root=Path(data_root))


def _panels_root(config: PipelineConfig) -> Path:
    """Where the factor store lives for this run's data root."""
    return config.data_root / "factors" / "panels"


# ---------------------------------------------------------------------------
# The summary -- the only rendering of an outcome
# ---------------------------------------------------------------------------
def format_summary(outcome: StrategyRunOutcome) -> str:
    """Render ``outcome`` as the command's stable, one-field-per-line summary.

    The text is a reporting surface, never a storage format: every number it prints
    is already in ``outcome`` at full precision, and the frame it is derived from is
    a copy, so printing can never change the result.  The metadata block is always
    present and always in the same order, followed by one ``required_factor`` line per
    exact ``(name, version)`` pair, then ``selection:``, the column header and one
    line per selected symbol in rank order (or :data:`EMPTY_SELECTION_TEXT`).

    Parameters
    ----------
    outcome:
        The run to render.

    Returns
    -------
    str
        The summary, without a trailing newline (``print`` adds one).
    """
    lines = [
        f"strategy_name: {outcome.strategy_name}",
        f"strategy_version: {outcome.strategy_version}",
        f"as_of_date: {outcome.as_of_date.isoformat()}",
        f"universe_size: {outcome.universe_size}",
        f"selected_count: {outcome.selected_count}",
    ]
    lines.extend(f"required_factor: {requirement}" for requirement in outcome.factor_requirements)
    lines.append("selection:")
    lines.append(_selection_header())

    selections = outcome.selections
    if not selections:
        lines.append(EMPTY_SELECTION_TEXT)
    else:
        lines.extend(_selection_line(selection) for selection in selections)

    return "\n".join(lines)


def _selection_header() -> str:
    """The column header of the selection table, in the same widths as the rows."""
    return f"{'rank':<{RANK_WIDTH}}{'symbol':<{SYMBOL_WIDTH}}score"


def _selection_line(selection: Selection) -> str:
    """One table row: rank, symbol and the score at :data:`SCORE_DECIMALS` decimals."""
    return (
        f"{selection.rank:<{RANK_WIDTH}}"
        f"{selection.symbol:<{SYMBOL_WIDTH}}"
        f"{selection.score:.{SCORE_DECIMALS}f}"
    )


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The argument parser of ``python scripts/run_strategy.py``.

    One strategy, one date: every flag below chooses *which* rule and *which* factor
    version it reads, *which* session it decides on, *where* the stored panels live or
    *which* pool it selects from.  There is deliberately no date range, no batch, no
    frequency, no anchor, no cache flag, no result file and no backtest switch; a
    missing, malformed or unknown flag is argparse's own exit code ``2``.
    """
    return _parser_for(PipelineConfig().data_root)


def _parser_for(default_data_root: Path) -> argparse.ArgumentParser:
    """Build the parser, naming the defaults the run will actually use."""
    parser = argparse.ArgumentParser(
        prog="run_strategy.py",
        description=(
            "Run one Top N strategy for one decision date from the stored factor panels "
            "and print the selected symbols."
        ),
    )
    parser.add_argument(
        "--factor",
        required=True,
        metavar="FACTOR_NAME",
        help="factor the strategy ranks by, e.g. momentum_12_1",
    )
    parser.add_argument(
        "--factor-version",
        dest="factor_version",
        required=True,
        metavar="VERSION",
        help=(
            "exact factor version to read, e.g. 1.0.0 (required: a selection is only "
            "traceable if the panel it came from is named)"
        ),
    )
    parser.add_argument(
        "--top-n",
        dest="top_n",
        required=True,
        type=int,
        metavar="N",
        help="how many rankable symbols to select (a positive integer)",
    )
    parser.add_argument(
        "--as-of",
        dest="as_of",
        required=True,
        metavar="YYYY-MM-DD",
        help=(
            "decision date as strict ISO-8601; its panel must already be stored, and the "
            "date is never adjusted to a neighbouring session"
        ),
    )
    parser.add_argument(
        "--strategy-version",
        dest="strategy_version",
        metavar="VERSION",
        default=DEFAULT_STRATEGY_VERSION,
        help=f"version recorded in the strategy's identity (default: {DEFAULT_STRATEGY_VERSION})",
    )
    parser.add_argument(
        "--data-root",
        dest="data_root",
        metavar="PATH",
        default=None,
        help=f"root holding factors/panels (default: {default_data_root})",
    )
    parser.add_argument(
        "--universe-path",
        dest="universe_path",
        metavar="PATH",
        default=None,
        help="static stock-pool CSV (default: the committed config/universe_seed.csv)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv``, run the strategy, print the selection and return the exit code.

    Parameters
    ----------
    argv:
        Command-line arguments without the program name (``None`` means
        ``sys.argv[1:]``).  Passing a list lets a test drive the command exactly as a
        shell would, without starting a second process.

    Returns
    -------
    int
        ``0`` when the strategy ran and the selection was printed -- including an empty
        selection, which is a fact about the cross-section, not a failure.  ``1`` when a
        :class:`~strategy_sys.errors.StrategyError`, a
        :class:`~factor_sys.errors.FactorError` or this module's
        :class:`RunStrategyError` stopped the run; that message goes to stderr as one
        ``error: ...`` line, never as a traceback.  argparse keeps its own ``2``.
    """
    args = build_parser().parse_args(argv)

    try:
        outcome = run_strategy(
            factor_name=args.factor,
            factor_version=args.factor_version,
            top_n=args.top_n,
            as_of_date=args.as_of,
            strategy_version=args.strategy_version,
            data_root=args.data_root,
            universe_path=args.universe_path,
        )
    except (FactorError, StrategyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_summary(outcome))
    return 0


__all__ = [
    "DEFAULT_STRATEGY_VERSION",
    "EMPTY_SELECTION_TEXT",
    "RunStrategyError",
    "Selection",
    "StrategyRunOutcome",
    "build_parser",
    "format_summary",
    "main",
    "run_strategy",
]


if __name__ == "__main__":
    raise SystemExit(main())
