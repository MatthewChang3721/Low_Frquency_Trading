"""Tests for the generic single-factor Top-N rule.

Everything here is in memory and offline: a ``UniverseSnapshot`` built from a
tuple of tickers, ``FactorPanel`` objects built from dicts, and a
``StrategyContext`` assembled around them.  No ``data/`` directory, factor store,
trading calendar or network call is involved, which is exactly the claim the
strategy makes -- it is a pure function from a context to a result.

Covered in this order:

* the rule's identity: name, version, required factor, read-only state,
* the selection itself: scores, ranks, tie-break, row order, ``top_n`` variants,
* symbols that cannot be ranked, and the legal empty selection,
* the context contract: a missing factor, an extra panel, exact identity,
* purity: the context, the panel and the caller's frame are never modified,
* the returned ``StrategyResult``: the spec it carries, identity, columns, column
  types,
* every rejected construction argument,
* the architectural boundary of the package and its import surface.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

import strategy_sys
import strategy_sys.strategies as strategies_module
from factor_sys import FactorPanel, UniverseSnapshot
from strategy_sys import (
    FORBIDDEN_RESULT_COLUMNS,
    STRATEGY_RESULT_COLUMNS,
    FactorRequirement,
    Strategy,
    StrategyContext,
    StrategyContractError,
    StrategyError,
    StrategyResult,
    StrategySpec,
)
from strategy_sys.strategies import TopNByFactorStrategy, single_factor
from strategy_sys.strategies.single_factor import top_n as top_n_module

#: The in-memory universe most fixtures below work with.
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")

#: The decision date of every fixture.
AS_OF: date = date(2024, 1, 5)

#: Exact factor identities: the one the fixtures produce, and two others.
MOMENTUM = FactorRequirement(factor_name="momentum_12_1", factor_version="1.0.0")
MOMENTUM_V2 = FactorRequirement(factor_name="momentum_12_1", factor_version="2.0.0")
VALUE = FactorRequirement(factor_name="book_to_price", factor_version="2.0.0")


def make_universe(symbols: Sequence[str] = SYMBOLS) -> UniverseSnapshot:
    """A validated pool for :data:`AS_OF` -- the only universe a context accepts."""
    return UniverseSnapshot(
        as_of_date=AS_OF,
        symbols=tuple(symbols),
        source="config/universe_seed.csv",
    )


def make_panel(
    requirement: FactorRequirement = MOMENTUM,
    symbols: Sequence[str] = SYMBOLS,
    raw_value: Sequence[float] = (0.20, 0.50, 0.10),
    is_eligible: Sequence[bool] | None = None,
) -> FactorPanel:
    """One validated cross-section of ``requirement``, in ``symbols`` order.

    ``is_eligible=None`` means "every symbol is eligible", so a test that only
    cares about a pool of another size does not have to repeat the flags.
    """
    eligibility = [True] * len(symbols) if is_eligible is None else list(is_eligible)
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(AS_OF)] * len(symbols),
            "symbol": list(symbols),
            "factor_name": [requirement.factor_name] * len(symbols),
            "factor_version": [requirement.factor_version] * len(symbols),
            "raw_value": list(raw_value),
            "is_eligible": eligibility,
        }
    )
    return FactorPanel(frame, universe=list(symbols), as_of_date=AS_OF)


def make_context(
    symbols: Sequence[str] = SYMBOLS,
    raw_value: Sequence[float] = (0.20, 0.50, 0.10),
    is_eligible: Sequence[bool] | None = None,
    *,
    factors: Sequence[FactorRequirement] = (MOMENTUM,),
) -> StrategyContext:
    """A context holding one identical-value panel per requirement in ``factors``."""
    panels = {
        requirement: make_panel(requirement, symbols, raw_value, is_eligible)
        for requirement in factors
    }
    return StrategyContext(
        as_of_date=AS_OF,
        universe=make_universe(symbols),
        factor_panels=panels,
    )


# ---------------------------------------------------------------------------
# The rule's identity
# ---------------------------------------------------------------------------
def test_the_strategy_declares_its_identity() -> None:
    strategy = TopNByFactorStrategy(MOMENTUM, 10)

    assert isinstance(strategy, Strategy)
    assert isinstance(strategy.spec, StrategySpec)
    assert strategy.spec.name == "top_10_by_momentum_12_1"
    assert strategy.spec.version == "1.0.0"
    assert strategy.spec.required_factors == (MOMENTUM,)
    assert strategy.factor_requirement is MOMENTUM
    assert strategy.top_n == 10
    assert strategy.version == "1.0.0"


@pytest.mark.parametrize(
    ("requirement", "top_n", "expected"),
    [
        (MOMENTUM, 10, "top_10_by_momentum_12_1"),
        (MOMENTUM, 1, "top_1_by_momentum_12_1"),
        (MOMENTUM, 50, "top_50_by_momentum_12_1"),
        (VALUE, 10, "top_10_by_book_to_price"),
    ],
)
def test_the_name_carries_the_size_and_the_factor(
    requirement: FactorRequirement,
    top_n: int,
    expected: str,
) -> None:
    """``Top 10`` and ``Top 50`` of one factor must not share an identity."""
    strategy = TopNByFactorStrategy(requirement, top_n)

    assert strategy.spec.name == expected
    assert strategy.spec.required_factors == (requirement,)
    assert strategy.top_n == top_n


def test_two_sizes_of_one_factor_are_two_strategy_identities() -> None:
    assert (
        TopNByFactorStrategy(MOMENTUM, 10).spec.name != TopNByFactorStrategy(MOMENTUM, 50).spec.name
    )


def test_the_version_is_the_one_the_caller_gave() -> None:
    """A new rule -- a new tie-break, a new direction -- is a new version."""
    strategy = TopNByFactorStrategy(MOMENTUM, 10, version="1.4.0")

    assert strategy.spec.version == "1.4.0"
    assert strategy.version == "1.4.0"


def test_the_rule_is_generic_and_not_a_momentum_wrapper() -> None:
    momentum = TopNByFactorStrategy(MOMENTUM, 10)
    value = TopNByFactorStrategy(VALUE, 10)

    assert type(momentum) is type(value) is TopNByFactorStrategy
    assert momentum.spec.required_factors != value.spec.required_factors
    assert not hasattr(top_n_module, "MomentumTopNStrategy")


def test_the_identity_is_stable_and_read_only() -> None:
    strategy = TopNByFactorStrategy(MOMENTUM, 10)

    assert strategy.spec is strategy.spec
    for name in ("spec", "top_n", "version", "factor_requirement"):
        with pytest.raises(AttributeError):
            setattr(strategy, name, None)


def test_the_size_is_a_required_argument() -> None:
    with pytest.raises(TypeError):
        TopNByFactorStrategy(MOMENTUM)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# The selection itself
# ---------------------------------------------------------------------------
def test_the_documented_example_selects_the_top_two() -> None:
    """One row per universe symbol, with the score, rank and flag of each."""
    symbols = ("MSFT", "AAPL", "NVDA", "TSLA")
    context = make_context(
        symbols,
        raw_value=(0.20, 0.20, 0.10, float("nan")),
        is_eligible=(True, True, True, False),
    )

    result = TopNByFactorStrategy(MOMENTUM, 2).compute(context)

    frame = result.frame
    assert tuple(frame.columns) == STRATEGY_RESULT_COLUMNS
    assert frame["symbol"].tolist() == list(symbols)
    assert frame["score"].tolist()[:3] == [0.20, 0.20, 0.10]
    assert pd.isna(frame["score"].tolist()[3])
    assert frame["rank"].tolist()[:3] == [2, 1, 3]
    assert frame["rank"].tolist()[3] is pd.NA
    assert frame["is_selected"].tolist() == [True, True, False, False]
    assert result.selected_symbols == ("MSFT", "AAPL")
    assert len(result) == len(symbols)


def test_the_rows_stay_in_universe_order() -> None:
    """The frame is never sorted: ``rank`` and ``is_selected`` carry the order."""
    symbols = ("TSLA", "AAPL", "MSFT")
    context = make_context(symbols, raw_value=(0.10, 0.30, 0.20))

    result = TopNByFactorStrategy(MOMENTUM, 3).compute(context)

    assert result.frame["symbol"].tolist() == ["TSLA", "AAPL", "MSFT"]
    assert result.frame["rank"].tolist() == [3, 1, 2]
    assert result.frame["score"].tolist() == [0.10, 0.30, 0.20]
    # ... which is neither the score order nor the alphabetical order.
    assert result.frame["symbol"].tolist() != ["AAPL", "MSFT", "TSLA"]


def test_scores_rank_from_the_highest() -> None:
    """A negative value is a score, not a missing one."""
    context = make_context(SYMBOLS, raw_value=(-0.20, 0.40, 0.00))

    result = TopNByFactorStrategy(MOMENTUM, 2).compute(context)

    assert result.frame["score"].tolist() == [-0.20, 0.40, 0.00]
    assert result.frame["rank"].tolist() == [3, 1, 2]
    assert result.frame["is_selected"].tolist() == [False, True, True]


def test_an_equal_score_breaks_the_tie_by_symbol() -> None:
    symbols = ("NVDA", "AAPL", "MSFT")
    context = make_context(symbols, raw_value=(0.50, 0.50, 0.50))

    result = TopNByFactorStrategy(MOMENTUM, 2).compute(context)

    # AAPL < MSFT < NVDA, so the plateau is ranked by ticker, not by row order.
    assert result.frame["rank"].tolist() == [3, 1, 2]
    assert result.frame["is_selected"].tolist() == [False, True, True]
    assert sorted(result.frame["rank"].tolist()) == [1, 2, 3]


def test_ranks_stay_consecutive_and_unique_across_plateaus() -> None:
    symbols = ("AAPL", "MSFT", "NVDA", "TSLA")
    context = make_context(symbols, raw_value=(0.50, 0.20, 0.50, 0.20))

    result = TopNByFactorStrategy(MOMENTUM, 4).compute(context)

    assert result.frame["rank"].tolist() == [1, 3, 2, 4]
    assert sorted(result.frame["rank"].tolist()) == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("top_n", "expected"),
    [
        (1, ("MSFT",)),
        (3, ("AAPL", "MSFT", "NVDA")),
        (99, ("AAPL", "MSFT", "NVDA")),
    ],
)
def test_top_n_variants(top_n: int, expected: tuple[str, ...]) -> None:
    """``top_n`` at the edge, exactly enough, and more than the pool holds."""
    result = TopNByFactorStrategy(MOMENTUM, top_n).compute(make_context())

    assert result.selected_symbols == expected
    assert len(result.selected_symbols) == min(top_n, len(SYMBOLS))


def test_a_selection_needs_a_finite_score_and_a_rank() -> None:
    """Every selected row is backed by a number and a position, as the contract asks."""
    result = TopNByFactorStrategy(MOMENTUM, 2).compute(make_context())

    selected = result.frame[result.frame["is_selected"]]
    assert len(selected) == 2
    assert selected["score"].notna().all()
    assert selected["rank"].notna().all()
    # In universe order, the two selections carry exactly the ranks 1 and 2.
    assert sorted(selected["rank"].tolist()) == [1, 2]


# ---------------------------------------------------------------------------
# Symbols that cannot be ranked
# ---------------------------------------------------------------------------
def test_an_ineligible_symbol_is_never_ranked_or_selected() -> None:
    context = make_context(SYMBOLS, raw_value=(0.90, 0.10, 0.50), is_eligible=(False, True, True))

    result = TopNByFactorStrategy(MOMENTUM, 99).compute(context)

    frame = result.frame
    assert pd.isna(frame.loc[0, "score"])
    assert frame.loc[0, "rank"] is pd.NA
    assert not bool(frame.loc[0, "is_selected"])
    assert frame["rank"].tolist()[1:] == [2, 1]
    assert result.selected_symbols == ("MSFT", "NVDA")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_value_is_not_ranked(value: float) -> None:
    context = make_context(SYMBOLS, raw_value=(value, 0.40, 0.20))

    result = TopNByFactorStrategy(MOMENTUM, 99).compute(context)

    frame = result.frame
    assert pd.isna(frame.loc[0, "score"])
    assert frame.loc[0, "rank"] is pd.NA
    assert not bool(frame.loc[0, "is_selected"])
    assert frame["rank"].tolist()[1:] == [1, 2]
    assert result.selected_symbols == ("MSFT", "NVDA")


def test_an_unrankable_symbol_keeps_its_row() -> None:
    """A symbol without a score is still part of the cross-section."""
    context = make_context(SYMBOLS, raw_value=(0.30, float("nan"), 0.10))

    result = TopNByFactorStrategy(MOMENTUM, 10).compute(context)

    assert result.frame["symbol"].tolist() == list(SYMBOLS)
    assert len(result) == len(SYMBOLS)
    assert result.frame["is_selected"].tolist() == [True, False, True]


def test_nothing_rankable_still_returns_a_legal_result() -> None:
    """An empty selection is a fact, not an error: the portfolio layer decides."""
    context = make_context(
        SYMBOLS,
        raw_value=(float("nan"), float("nan"), float("nan")),
        is_eligible=(False, False, False),
    )

    result = TopNByFactorStrategy(MOMENTUM, 10).compute(context)

    assert isinstance(result, StrategyResult)
    assert len(result) == len(SYMBOLS)
    assert result.selected_symbols == ()
    assert result.frame["rank"].isna().all()
    assert result.frame["score"].isna().all()
    assert not result.frame["is_selected"].any()
    assert result.frame["symbol"].tolist() == list(SYMBOLS)


# ---------------------------------------------------------------------------
# The context contract
# ---------------------------------------------------------------------------
def test_a_missing_factor_names_what_it_needs_and_what_it_has() -> None:
    context = make_context(factors=(VALUE,))

    with pytest.raises(StrategyContractError) as excinfo:
        TopNByFactorStrategy(MOMENTUM, 10).compute(context)

    message = str(excinfo.value)
    assert "top_10_by_momentum_12_1" in message
    assert "momentum_12_1/1.0.0" in message
    assert "book_to_price/2.0.0" in message
    assert "no partial result" in message
    assert excinfo.value.problems


def test_the_panel_is_matched_by_exact_identity() -> None:
    """The same factor name at another version is another factor."""
    context = make_context(factors=(MOMENTUM_V2,))

    with pytest.raises(StrategyContractError) as excinfo:
        TopNByFactorStrategy(MOMENTUM, 10).compute(context)

    assert "momentum_12_1/1.0.0" in str(excinfo.value)
    assert "momentum_12_1/2.0.0" in str(excinfo.value)


def test_an_unrelated_extra_panel_changes_nothing() -> None:
    alone = TopNByFactorStrategy(MOMENTUM, 2).compute(make_context())
    with_extra = TopNByFactorStrategy(MOMENTUM, 2).compute(make_context(factors=(MOMENTUM, VALUE)))

    pd.testing.assert_frame_equal(alone.frame, with_extra.frame)


def test_a_second_strategy_reads_its_own_factor() -> None:
    context = make_context(factors=(MOMENTUM, VALUE))

    result = TopNByFactorStrategy(VALUE, 1).compute(context)

    assert result.strategy_name == "top_1_by_book_to_price"
    assert result.selected_symbols == ("MSFT",)


def test_compute_requires_a_context() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        TopNByFactorStrategy(MOMENTUM, 10).compute(object())  # type: ignore[arg-type]

    assert "StrategyContext" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Purity: nothing the caller owns is touched
# ---------------------------------------------------------------------------
def test_computing_does_not_touch_the_context() -> None:
    context = make_context()
    panel = context.factor_panels[MOMENTUM]
    panel_rows = panel.frame
    requirements = list(context.factor_panels)

    TopNByFactorStrategy(MOMENTUM, 2).compute(context)

    assert list(context.factor_panels) == requirements
    assert context.factor_panels[MOMENTUM] is panel
    assert context.as_of_date == AS_OF
    assert context.universe.symbols == SYMBOLS
    pd.testing.assert_frame_equal(panel.frame, panel_rows)


def test_computing_does_not_touch_the_callers_frame() -> None:
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(AS_OF)] * 3,
            "symbol": list(SYMBOLS),
            "factor_name": ["momentum_12_1"] * 3,
            "factor_version": ["1.0.0"] * 3,
            "raw_value": [0.20, 0.50, 0.10],
            "is_eligible": [True, True, True],
        }
    )
    written = frame.copy(deep=True)
    panel = FactorPanel(frame, universe=SYMBOLS, as_of_date=AS_OF)
    panel_rows = panel.frame
    context = StrategyContext(
        as_of_date=AS_OF,
        universe=make_universe(),
        factor_panels={MOMENTUM: panel},
    )

    TopNByFactorStrategy(MOMENTUM, 2).compute(context)

    pd.testing.assert_frame_equal(frame, written)
    pd.testing.assert_frame_equal(panel.frame, panel_rows)


def test_repeated_calls_are_equal_but_separate() -> None:
    strategy = TopNByFactorStrategy(MOMENTUM, 2)
    context = make_context()

    first = strategy.compute(context)
    second = strategy.compute(context)

    assert first is not second
    pd.testing.assert_frame_equal(first.frame, second.frame)

    changed = first.frame
    changed.loc[2, "is_selected"] = True
    assert bool(changed.loc[2, "is_selected"])
    assert not bool(second.frame.loc[2, "is_selected"])


def test_the_result_frame_is_a_defensive_copy() -> None:
    result = TopNByFactorStrategy(MOMENTUM, 1).compute(make_context())

    frame = result.frame
    frame.loc[0, "score"] = 99.0
    frame.loc[0, "is_selected"] = True

    assert result.frame.loc[0, "score"] == 0.20
    assert bool(result.frame.loc[0, "is_selected"]) is False


def test_the_panel_frame_is_a_copy_of_the_panel_not_the_panel() -> None:
    """Reading the panel twice yields equal, independent frames."""
    context = make_context()
    panel = context.factor_panels[MOMENTUM]

    first = panel.frame
    second = panel.frame

    assert first is not second
    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# The returned StrategyResult
# ---------------------------------------------------------------------------
def test_the_result_identity_comes_from_the_spec_and_the_context() -> None:
    strategy = TopNByFactorStrategy(MOMENTUM, 2, version="1.4.0")
    context = make_context()

    result = strategy.compute(context)

    assert isinstance(result, StrategyResult)
    assert result.as_of_date == AS_OF
    assert isinstance(result.as_of_date, date)
    assert not isinstance(result.as_of_date, datetime)
    assert result.strategy_spec is strategy.spec
    assert result.strategy_name == strategy.spec.name == "top_2_by_momentum_12_1"
    assert result.strategy_version == strategy.spec.version == "1.4.0"
    assert result.universe is context.universe
    assert tuple(result.frame.columns) == STRATEGY_RESULT_COLUMNS
    assert result.frame["as_of_date"].tolist() == [pd.Timestamp(AS_OF)] * len(SYMBOLS)


def test_the_result_carries_the_complete_factor_identity() -> None:
    """The dependency travels as metadata: name *and* version, not just a name."""
    strategy = TopNByFactorStrategy(MOMENTUM, 2)

    result = strategy.compute(make_context())

    assert result.factor_requirements == (FactorRequirement("momentum_12_1", "1.0.0"),)
    assert result.factor_requirements is strategy.spec.required_factors
    assert result.factor_requirements == result.strategy_spec.required_factors
    assert isinstance(result.factor_requirements, tuple)
    assert tuple(str(requirement) for requirement in result.factor_requirements) == (
        "momentum_12_1/1.0.0",
    )


def test_the_result_adds_no_factor_column() -> None:
    """A dependency is a property of the cross-section, not of a symbol."""
    frame = TopNByFactorStrategy(MOMENTUM, 2).compute(make_context()).frame

    assert tuple(frame.columns) == STRATEGY_RESULT_COLUMNS
    assert "factor_name" not in frame.columns
    assert "factor_version" not in frame.columns
    assert len(frame) == len(SYMBOLS)


def test_two_factor_versions_of_the_same_strategy_are_distinguishable() -> None:
    """Same strategy name and version, different factor version -- still traceable.

    The ranking rule is unchanged, so the two results share ``strategy_name`` and
    ``strategy_version`` and even the same frame; the attached spec is what tells
    a later portfolio, backtest or cache which factor version produced them.
    """
    on_v1 = TopNByFactorStrategy(MOMENTUM, 2, version="1.0.0")
    on_v2 = TopNByFactorStrategy(MOMENTUM_V2, 2, version="1.0.0")

    result_v1 = on_v1.compute(make_context())
    result_v2 = on_v2.compute(make_context(factors=(MOMENTUM_V2,)))

    assert result_v1.strategy_name == result_v2.strategy_name
    assert result_v1.strategy_version == result_v2.strategy_version == "1.0.0"
    assert result_v1.factor_requirements != result_v2.factor_requirements
    assert result_v1.strategy_spec != result_v2.strategy_spec
    assert result_v1.factor_requirements == (MOMENTUM,)
    assert result_v2.factor_requirements == (MOMENTUM_V2,)
    pd.testing.assert_frame_equal(result_v1.frame, result_v2.frame)


def test_the_result_column_types() -> None:
    frame = TopNByFactorStrategy(MOMENTUM, 2).compute(make_context()).frame

    assert str(frame["score"].dtype) == "float64"
    assert str(frame["rank"].dtype) == "Int64"
    assert str(frame["is_selected"].dtype) == "bool"


def test_the_result_carries_no_portfolio_column() -> None:
    """A weight, a quantity, an order or a fill belongs to a later layer."""
    frame = TopNByFactorStrategy(MOMENTUM, 2).compute(make_context()).frame

    assert set(frame.columns) & set(FORBIDDEN_RESULT_COLUMNS) == set()
    for column in ("weight", "quantity", "order", "side", "fill_price", "cash"):
        assert column not in frame.columns


# ---------------------------------------------------------------------------
# Rejected construction arguments
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "requirement",
    ["momentum_12_1", ("momentum_12_1", "1.0.0"), ["momentum_12_1", "1.0.0"], None, 10],
)
def test_construction_refuses_something_that_is_not_a_requirement(requirement: object) -> None:
    """A name without a version is not an identity, so it is refused."""
    with pytest.raises(StrategyContractError) as excinfo:
        TopNByFactorStrategy(requirement, 10)  # type: ignore[arg-type]

    assert "factor_requirement must be a FactorRequirement" in str(excinfo.value)


@pytest.mark.parametrize("top_n", [0, -1, -50, 1.0, 10.0, "10", True, False, None])
def test_construction_refuses_a_bad_size(top_n: object) -> None:
    """Nothing is converted: ``"10"``, ``10.0`` and ``True`` are not ``10``."""
    with pytest.raises(StrategyContractError) as excinfo:
        TopNByFactorStrategy(MOMENTUM, top_n)  # type: ignore[arg-type]

    assert "top_n" in str(excinfo.value)


def test_the_size_message_says_why_it_is_not_converted() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        TopNByFactorStrategy(MOMENTUM, "10")  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "refused rather than converted" in message
    assert "got str" in message


def test_a_zero_size_is_refused_in_its_own_words() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        TopNByFactorStrategy(MOMENTUM, 0)

    assert "top_n must be greater than zero" in str(excinfo.value)


@pytest.mark.parametrize("version", ["", "   ", "\t", None, 10, 1.0])
def test_construction_refuses_a_bad_version(version: object) -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        TopNByFactorStrategy(MOMENTUM, 10, version=version)  # type: ignore[arg-type]

    assert "version must be a non-empty string" in str(excinfo.value)


def test_every_construction_problem_is_reported_at_once() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        TopNByFactorStrategy("momentum_12_1", 0, version="")  # type: ignore[arg-type]

    problems = excinfo.value.problems
    assert len(problems) == 3
    joined = "\n".join(problems)
    assert "factor_requirement" in joined
    assert "top_n" in joined
    assert "version" in joined
    assert str(excinfo.value).startswith("invalid TopNByFactorStrategy\n  - ")


def test_a_rejected_construction_is_a_strategy_error() -> None:
    with pytest.raises(StrategyError):
        TopNByFactorStrategy(None, 10)  # type: ignore[arg-type]


def test_an_integer_valued_size_is_stored_as_an_int() -> None:
    """A plain ``int`` passes untouched; nothing else is coerced on the way in."""
    strategy = TopNByFactorStrategy(MOMENTUM, 7)

    assert strategy.top_n == 7
    assert isinstance(strategy.top_n, int)


# ---------------------------------------------------------------------------
# Architectural boundary
# ---------------------------------------------------------------------------
#: Modules the strategies may import: the standard library they need, pandas,
#: and the strategy layer's own contracts -- nothing else.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "numbers",
        "pandas",
        "strategy_sys.contracts",
        "strategy_sys.errors",
        "strategy_sys.strategies",
        "strategy_sys.strategies.single_factor",
        "strategy_sys.strategies.single_factor.top_n",
    }
)

#: Text that must never appear: storage, market data, calendars, factor plumbing,
#: portfolio construction, the network or the file system.
FORBIDDEN_TEXT: tuple[str, ...] = (
    "from data_sys",
    "import data_sys",
    "from factor_sys",
    "factor_sys.runner",
    "factor_sys.factor_store",
    "factor_sys.registry",
    "factors.momentum",
    "FactorStore",
    "FactorRunner",
    "pyarrow",
    "duckdb",
    "yfinance",
    "requests",
    "httpx",
    "socket",
    "exchange_calendars",
    "market_calendar",
    "read_parquet",
    "to_parquet",
    "urlopen",
    "pathlib",
)


def strategy_modules() -> list[Path]:
    """The three modules this task added to the strategy package."""
    package = Path(strategy_sys.__file__).parent / "strategies"
    return [
        package / "__init__.py",
        package / "single_factor" / "__init__.py",
        package / "single_factor" / "top_n.py",
    ]


def imported_modules(path: Path) -> set[str]:
    """Every module named by an import statement in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_the_strategy_imports_only_its_own_layers() -> None:
    for module in strategy_modules():
        imported = imported_modules(module)
        assert imported <= ALLOWED_IMPORTS, (module.name, sorted(imported - ALLOWED_IMPORTS))


def non_doctest_text(path: Path) -> str:
    """The module's source without its doctest examples.

    A docstring may *demonstrate* how a caller assembles a context -- the runner
    builds one out of factor-layer objects -- while the module itself must never
    import that layer.  Doctest lines are therefore excluded here; the import
    check above stays the authoritative boundary test.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.strip().startswith((">>>", "...")))


def test_the_strategy_never_names_a_forbidden_dependency() -> None:
    for module in strategy_modules():
        source = non_doctest_text(module)
        offenders = [text for text in FORBIDDEN_TEXT if text in source]
        assert offenders == [], (module.name, offenders)


def test_the_strategy_never_opens_prints_or_executes_anything() -> None:
    """No file, no console, no ``eval``: the rule is arithmetic and sorting."""
    for module in strategy_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        called = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        assert called & {"open", "print", "input", "eval", "exec", "compile"} == set(), module.name


# ---------------------------------------------------------------------------
# Import surface
# ---------------------------------------------------------------------------
def test_both_import_paths_reach_the_same_class() -> None:
    from strategy_sys.strategies import TopNByFactorStrategy as via_family
    from strategy_sys.strategies.single_factor import TopNByFactorStrategy as via_rule

    assert via_family is via_rule is TopNByFactorStrategy


def test_the_packages_export_the_strategy() -> None:
    assert strategies_module.__all__ == ["TopNByFactorStrategy"]
    assert single_factor.__all__ == ["DEFAULT_VERSION", "TopNByFactorStrategy"]
    assert top_n_module.__all__ == ["DEFAULT_VERSION", "TopNByFactorStrategy"]
    assert single_factor.TopNByFactorStrategy is TopNByFactorStrategy
    assert top_n_module.TopNByFactorStrategy is TopNByFactorStrategy
    assert top_n_module.DEFAULT_VERSION == "1.0.0"


def test_the_package_root_stays_the_contract() -> None:
    """No concrete rule at the root: a rule is reached through its own module."""
    assert not hasattr(strategy_sys, "TopNByFactorStrategy")
    assert "TopNByFactorStrategy" not in strategy_sys.__all__
    for absent in ("MomentumTopNStrategy", "ValueTopNStrategy"):
        assert not hasattr(strategy_sys, absent)
    # The runner is layer plumbing rather than a rule -- it takes the strategy it
    # runs as an argument and imports no rule (checked in test_strategy_runner.py).
    assert strategy_sys.StrategyRunner.__module__ == "strategy_sys.runner"
    assert not hasattr(strategy_sys, "run_strategy")
