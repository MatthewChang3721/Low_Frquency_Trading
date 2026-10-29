"""Tests for the strategy data contract (:mod:`strategy_sys.contracts`).

Everything here is in memory and offline: the universe is a
``UniverseSnapshot`` built from a tuple of tickers, every factor panel is built
with ``FactorPanel``, and no ``data/`` directory, factor store, trading calendar
or network call is involved.  That is exactly the shape the future strategy
runner will hand to a strategy, so these tests pin down the shared vocabulary
before any strategy exists.

Covered in this order:

* the abstract :class:`Strategy` interface, and a minimal strategy that uses it,
* ``FactorRequirement`` / ``StrategySpec``: identity, freezing, duplicates,
* ``StrategyContext``: several panels, dates, snapshot, symbol order, isolation,
* ``StrategyResult``: the authoritative spec it carries, round-trip, dtypes, row
  order, defensive copies, and every rejection the contract promises,
* collected problems: one error names them all,
* the architectural boundary of the package and its exports.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

import strategy_sys
import strategy_sys.contracts as contracts_module
from data_sys.errors import DataPipelineError
from factor_sys import FactorError, FactorPanel, UniverseSnapshot
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

#: The in-memory universe every fixture below works with.
SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")

#: The decision date of every fixture, and a different session to reject.
AS_OF: date = date(2024, 1, 5)
OTHER_SESSION: date = date(2024, 1, 8)

#: Two exact factor identities: one the fixtures produce, one they do not.
MOMENTUM = FactorRequirement(factor_name="momentum_12_1", factor_version="1.0.0")
VALUE = FactorRequirement(factor_name="book_to_price", factor_version="2.0.0")

#: The identity the fixtures write into a result.
STRATEGY_NAME = "top_n_momentum"
STRATEGY_VERSION = "1.0.0"

#: The spec the fixtures declare: the authoritative identity of a result.
SPEC = StrategySpec(
    name=STRATEGY_NAME,
    version=STRATEGY_VERSION,
    required_factors=(MOMENTUM,),
)

#: The same strategy rule, computed from another version of the same factor.
SPEC_ON_FACTOR_V2 = StrategySpec(
    name=STRATEGY_NAME,
    version=STRATEGY_VERSION,
    required_factors=(FactorRequirement(factor_name="momentum_12_1", factor_version="2.0.0"),),
)


# ---------------------------------------------------------------------------
# In-memory fixtures (no disk, no network, no store)
# ---------------------------------------------------------------------------
def make_universe(
    symbols: Sequence[str] = SYMBOLS,
    *,
    as_of_date: object = AS_OF,
    source: str = "config/universe_seed.csv",
) -> UniverseSnapshot:
    """A :class:`UniverseSnapshot` for the fixtures."""
    return UniverseSnapshot(as_of_date=as_of_date, symbols=tuple(symbols), source=source)  # type: ignore[arg-type]


def make_panel(
    *,
    symbols: Sequence[str] = SYMBOLS,
    as_of_date: object = AS_OF,
    factor_name: str = MOMENTUM.factor_name,
    factor_version: str = MOMENTUM.factor_version,
    raw_value: object = None,
    is_eligible: object = True,
) -> FactorPanel:
    """A valid :class:`FactorPanel` with one row per symbol."""
    values = [round(0.1 * (position + 1), 2) for position in range(len(symbols))]
    frame = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(as_of_date)] * len(symbols),  # type: ignore[arg-type]
            "symbol": list(symbols),
            "factor_name": [factor_name] * len(symbols),
            "factor_version": [factor_version] * len(symbols),
            "raw_value": values if raw_value is None else raw_value,
            "is_eligible": [is_eligible] * len(symbols),
        }
    )
    return FactorPanel(frame, universe=list(symbols), as_of_date=as_of_date)  # type: ignore[arg-type]


def make_context(
    panels: Mapping[FactorRequirement, FactorPanel] | None = None,
    *,
    as_of_date: object = AS_OF,
    universe: object = None,
) -> StrategyContext:
    """A :class:`StrategyContext` holding the momentum panel by default.

    The snapshot always keeps :data:`AS_OF`: these fixtures test the context, so
    an unusable ``as_of_date`` must reach the context rather than be refused by
    the snapshot first.
    """
    snapshot = make_universe() if universe is None else universe
    factor_panels = {MOMENTUM: make_panel()} if panels is None else panels
    return StrategyContext(
        as_of_date=as_of_date,  # type: ignore[arg-type]
        universe=snapshot,  # type: ignore[arg-type]
        factor_panels=factor_panels,
    )


def make_result_frame(
    symbols: Sequence[str] = SYMBOLS,
    *,
    as_of_date: object = AS_OF,
    strategy_name: object = STRATEGY_NAME,
    strategy_version: object = STRATEGY_VERSION,
    score: object = None,
    rank: object = None,
    is_selected: object = None,
) -> pd.DataFrame:
    """A complete, valid long-format strategy result."""
    count = len(symbols)
    scores = [0.9 - 0.4 * position for position in range(count)]
    ranks = list(range(1, count + 1))
    selection = [True] + [False] * (count - 1)
    return pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp(as_of_date)] * count,  # type: ignore[arg-type]
            "strategy_name": [strategy_name] * count,
            "strategy_version": [strategy_version] * count,
            "symbol": list(symbols),
            "score": scores if score is None else score,
            "rank": ranks if rank is None else rank,
            "is_selected": selection if is_selected is None else is_selected,
        },
        columns=list(STRATEGY_RESULT_COLUMNS),
    )


def make_result(
    frame: object = None,
    *,
    universe: object = None,
    **options: object,
) -> StrategyResult:
    """A :class:`StrategyResult` built from in-memory output.

    ``strategy_spec`` defaults to :data:`SPEC` -- the identity
    :func:`make_result_frame` writes into the frame -- so a test only passes one
    when it is deliberately probing the spec itself.
    """
    snapshot = make_universe() if universe is None else universe
    rows = make_result_frame() if frame is None else frame
    options.setdefault("strategy_spec", SPEC)
    return StrategyResult(
        rows,  # type: ignore[arg-type]
        universe=snapshot,  # type: ignore[arg-type]
        **options,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Strategy: the abstract interface
# ---------------------------------------------------------------------------
class TopOneByFactorStrategy(Strategy):
    """The smallest useful strategy: score from one panel, select the best symbol.

    It is a test double, not a shipped strategy: it only exists to prove that the
    contract can express "score this cross-section and select" without weights,
    orders or a calendar.
    """

    spec = StrategySpec(
        name=STRATEGY_NAME,
        version=STRATEGY_VERSION,
        required_factors=(MOMENTUM,),
    )

    def compute(self, context: StrategyContext) -> StrategyResult:
        panel = context.factor_panels[self.spec.required_factors[0]]
        frame = panel.frame
        by_symbol = dict(zip(frame["symbol"], frame["raw_value"], strict=True))

        symbols = context.universe.symbols
        scored: list[float | None] = []
        for symbol in symbols:
            value = by_symbol.get(symbol)
            scored.append(None if value is None or pd.isna(value) else float(value))

        usable = [value for value in scored if value is not None]
        best = max(usable) if usable else None
        ranks: list[int | None] = [None] * len(scored)
        order = sorted(
            (index for index, value in enumerate(scored) if value is not None),
            key=lambda index: (-float(scored[index]), index),
        )
        for position, index in enumerate(order, start=1):
            ranks[index] = position

        return StrategyResult(
            pd.DataFrame(
                {
                    "as_of_date": [pd.Timestamp(context.as_of_date)] * len(symbols),
                    "strategy_name": [self.spec.name] * len(symbols),
                    "strategy_version": [self.spec.version] * len(symbols),
                    "symbol": list(symbols),
                    "score": pd.Series(scored, dtype="float64"),
                    "rank": pd.Series(ranks, dtype="Int64"),
                    "is_selected": [value is not None and value == best for value in scored],
                }
            ),
            universe=context.universe,
            strategy_spec=self.spec,
            as_of_date=context.as_of_date,
        )


def test_strategy_is_an_abstract_base_class() -> None:
    assert {"spec", "compute"} <= set(Strategy.__abstractmethods__)

    with pytest.raises(TypeError):
        Strategy()  # type: ignore[abstract]


def test_strategy_requires_both_spec_and_compute() -> None:
    class SpecOnly(Strategy):
        spec = StrategySpec(name="spec_only", version="1.0.0", required_factors=(MOMENTUM,))

    class ComputeOnly(Strategy):
        def compute(self, context: StrategyContext) -> StrategyResult:  # pragma: no cover
            raise NotImplementedError

    for incomplete in (SpecOnly, ComputeOnly):
        with pytest.raises(TypeError):
            incomplete()  # type: ignore[abstract]


def test_strategy_may_declare_its_spec_as_a_property_or_a_class_attribute() -> None:
    class PropertyStrategy(Strategy):
        @property
        def spec(self) -> StrategySpec:
            return StrategySpec(
                name="from_property",
                version="1.0.0",
                required_factors=(MOMENTUM,),
            )

        def compute(self, context: StrategyContext) -> StrategyResult:  # pragma: no cover
            raise NotImplementedError

    assert PropertyStrategy().spec.name == "from_property"
    assert TopOneByFactorStrategy().spec is TopOneByFactorStrategy.spec


def test_a_minimal_strategy_scores_and_selects_from_the_context() -> None:
    """The contract is enough for "score, rank and select" -- and nothing more."""
    strategy = TopOneByFactorStrategy()
    result = strategy.compute(make_context())

    assert isinstance(result, StrategyResult)
    assert result.as_of_date == AS_OF
    assert result.strategy_name == STRATEGY_NAME
    assert result.strategy_version == STRATEGY_VERSION
    assert result.strategy_spec is strategy.spec
    assert result.factor_requirements == (MOMENTUM,)
    assert result.symbols == SYMBOLS
    assert result.selected_symbols == ("NVDA",)
    assert result.frame["rank"].tolist() == [3, 2, 1]
    assert result.frame["score"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert repr(strategy).startswith("TopOneByFactorStrategy(spec=StrategySpec(")


def test_a_strategy_result_is_validated_by_the_contract() -> None:
    """A strategy that forgets a symbol cannot smuggle a broken result through."""

    class SloppyStrategy(Strategy):
        spec = StrategySpec(name="sloppy", version="1.0.0", required_factors=(MOMENTUM,))

        def compute(self, context: StrategyContext) -> StrategyResult:
            return StrategyResult(
                make_result_frame(SYMBOLS[:-1]),
                universe=context.universe,
                strategy_spec=self.spec,
            )

    with pytest.raises(StrategyContractError):
        SloppyStrategy().compute(make_context())


def test_the_context_does_not_police_a_spec_against_its_panels() -> None:
    """Matching ``spec.required_factors`` with the panels is the strategy's job."""
    context = make_context()
    assert set(TopOneByFactorStrategy().spec.required_factors) <= set(context.factor_panels)

    class NeedsValue(Strategy):
        spec = StrategySpec(name="needs_value", version="1.0.0", required_factors=(VALUE,))

        def compute(self, context: StrategyContext) -> StrategyResult:  # pragma: no cover
            missing = [
                str(requirement)
                for requirement in self.spec.required_factors
                if requirement not in context.factor_panels
            ]
            raise LookupError(f"missing factor panel(s): {missing}")

    with pytest.raises(LookupError, match="book_to_price/2.0.0"):
        NeedsValue().compute(context)


# ---------------------------------------------------------------------------
# FactorRequirement: one exact factor identity
# ---------------------------------------------------------------------------
def test_factor_requirement_holds_an_exact_identity() -> None:
    requirement = FactorRequirement(factor_name="momentum_12_1", factor_version="1.0.0")

    assert requirement.factor_name == "momentum_12_1"
    assert requirement.factor_version == "1.0.0"
    assert str(requirement) == "momentum_12_1/1.0.0"
    assert requirement == MOMENTUM
    assert requirement != FactorRequirement("momentum_12_1", "2.0.0")
    assert len({requirement, MOMENTUM, VALUE}) == 2


def test_factor_requirement_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        MOMENTUM.factor_version = "2.0.0"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("name", "version"),
    [
        ("", "1.0.0"),
        ("   ", "1.0.0"),
        ("Momentum", "1.0.0"),
        ("momentum-12-1", "1.0.0"),
        (None, "1.0.0"),
        ("momentum_12_1", ""),
        ("momentum_12_1", "   "),
        ("momentum_12_1", None),
    ],
)
def test_factor_requirement_rejects_an_unusable_identity(name: object, version: object) -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        FactorRequirement(factor_name=name, factor_version=version)  # type: ignore[arg-type]

    assert "FactorRequirement" in str(excinfo.value)
    assert excinfo.value.problems


def test_factor_requirement_reports_both_halves_at_once() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        FactorRequirement(factor_name="Momentum", factor_version="")

    assert len(excinfo.value.problems) == 2
    assert "factor_name" in str(excinfo.value)
    assert "factor_version" in str(excinfo.value)


# ---------------------------------------------------------------------------
# StrategySpec: what a strategy is and what it needs
# ---------------------------------------------------------------------------
def test_strategy_spec_accepts_a_single_factor_strategy() -> None:
    spec = StrategySpec(
        name=STRATEGY_NAME,
        version=STRATEGY_VERSION,
        required_factors=(MOMENTUM,),
    )

    assert spec.name == STRATEGY_NAME
    assert spec.version == STRATEGY_VERSION
    assert spec.required_factors == (MOMENTUM,)
    assert isinstance(spec.required_factors, tuple)


def test_strategy_spec_accepts_a_multi_factor_strategy() -> None:
    spec = StrategySpec(
        name="multi_factor_blend",
        version="0.2.0",
        required_factors=(MOMENTUM, VALUE),
    )

    assert [str(requirement) for requirement in spec.required_factors] == [
        "momentum_12_1/1.0.0",
        "book_to_price/2.0.0",
    ]


def test_strategy_spec_is_frozen_and_hashable() -> None:
    spec = StrategySpec(name=STRATEGY_NAME, version=STRATEGY_VERSION, required_factors=(MOMENTUM,))

    with pytest.raises(FrozenInstanceError):
        spec.version = "2.0.0"  # type: ignore[misc]
    assert len({spec, StrategySpec(STRATEGY_NAME, STRATEGY_VERSION, (MOMENTUM,))}) == 1


def test_strategy_spec_keeps_the_declared_factor_order() -> None:
    """The tuple is the order the strategy reads its panels in."""
    spec = StrategySpec(
        name=STRATEGY_NAME,
        version=STRATEGY_VERSION,
        required_factors=(VALUE, MOMENTUM),
    )

    assert spec.required_factors == (VALUE, MOMENTUM)
    assert isinstance(spec.required_factors, tuple)


@pytest.mark.parametrize(
    "required_factors",
    [
        [MOMENTUM],
        [MOMENTUM, VALUE],
        {MOMENTUM},
        {MOMENTUM, VALUE},
        (requirement for requirement in (MOMENTUM,)),
        iter([MOMENTUM]),
    ],
    ids=["list", "list-of-two", "set", "set-of-two", "generator", "iterator"],
)
def test_strategy_spec_rejects_anything_but_a_tuple(required_factors: object) -> None:
    """A list, a set or a generator cannot state a factor order."""
    with pytest.raises(StrategyContractError) as excinfo:
        StrategySpec(
            name=STRATEGY_NAME,
            version=STRATEGY_VERSION,
            required_factors=required_factors,  # type: ignore[arg-type]
        )

    message = str(excinfo.value)
    assert "required_factors" in message
    assert "tuple" in message


def test_strategy_spec_rejects_a_duplicated_requirement() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        StrategySpec(
            name=STRATEGY_NAME,
            version=STRATEGY_VERSION,
            required_factors=(MOMENTUM, FactorRequirement("momentum_12_1", "1.0.0")),
        )

    assert "momentum_12_1/1.0.0" in str(excinfo.value)
    assert "more than once" in str(excinfo.value)


@pytest.mark.parametrize(
    "required_factors",
    [
        (),
        MOMENTUM,
        "momentum_12_1",
        (MOMENTUM, "book_to_price"),
        (MOMENTUM, 42),
        ("momentum_12_1",),
    ],
)
def test_strategy_spec_rejects_vague_requirements(required_factors: object) -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        StrategySpec(
            name=STRATEGY_NAME,
            version=STRATEGY_VERSION,
            required_factors=required_factors,  # type: ignore[arg-type]
        )

    assert "required_factors" in str(excinfo.value)


@pytest.mark.parametrize(
    ("name", "version"),
    [
        ("", "1.0.0"),
        ("   ", "1.0.0"),
        ("Top N", "1.0.0"),
        ("top-n", "1.0.0"),
        (None, "1.0.0"),
        ("top_n_momentum", ""),
        ("top_n_momentum", "   "),
        ("top_n_momentum", None),
    ],
)
def test_strategy_spec_rejects_an_unusable_identity(name: object, version: object) -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        StrategySpec(
            name=name,  # type: ignore[arg-type]
            version=version,  # type: ignore[arg-type]
            required_factors=(MOMENTUM,),
        )

    assert "StrategySpec" in str(excinfo.value)


def test_strategy_spec_collects_every_problem() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        StrategySpec(name="", version="", required_factors=(MOMENTUM, 42))

    problems = excinfo.value.problems
    assert len(problems) == 3
    assert any("name" in problem for problem in problems)
    assert any("version" in problem for problem in problems)
    assert any("42" in problem for problem in problems)


def test_strategy_spec_has_no_strategy_parameters() -> None:
    """``top_n``, weights and cash belong to a strategy or the portfolio layer."""
    assert {record.name for record in fields(StrategySpec)} == {
        "name",
        "version",
        "required_factors",
    }


# ---------------------------------------------------------------------------
# StrategyContext: what a strategy may look at
# ---------------------------------------------------------------------------
def test_context_accepts_several_panels() -> None:
    value_panel = make_panel(factor_name=VALUE.factor_name, factor_version=VALUE.factor_version)
    universe = make_universe()
    context = StrategyContext(
        as_of_date=AS_OF,
        universe=universe,
        factor_panels={MOMENTUM: make_panel(), VALUE: value_panel},
    )

    assert context.as_of_date == AS_OF
    assert context.universe is universe
    assert list(context.factor_panels) == [MOMENTUM, VALUE]
    assert context.factor_panels[VALUE] is value_panel
    assert len(context.factor_panels) == 2


def test_context_accepts_a_date_or_a_plain_iso_string() -> None:
    for value in ("2024-01-05", " 2024-01-05 ", AS_OF, date(2024, 1, 5)):
        context = StrategyContext(
            as_of_date=value,  # type: ignore[arg-type]
            universe=make_universe(),
            factor_panels={MOMENTUM: make_panel()},
        )
        assert context.as_of_date == AS_OF


def test_context_copies_the_mapping_it_was_given() -> None:
    panels = {MOMENTUM: make_panel()}
    context = StrategyContext(
        as_of_date=AS_OF,
        universe=make_universe(),
        factor_panels=panels,
    )

    panels[VALUE] = make_panel(factor_name=VALUE.factor_name, factor_version=VALUE.factor_version)
    panels.clear()

    assert list(context.factor_panels) == [MOMENTUM]


def test_context_exposes_a_read_only_mapping() -> None:
    context = make_context()

    assert isinstance(context.factor_panels, Mapping)
    with pytest.raises(TypeError):
        context.factor_panels[VALUE] = make_panel()  # type: ignore[index]


def test_context_shares_the_panel_and_the_snapshot_it_was_given() -> None:
    """It copies the mapping, not the immutable objects inside it."""
    panel = make_panel()
    snapshot = make_universe()
    context = StrategyContext(
        as_of_date=AS_OF,
        universe=snapshot,
        factor_panels={MOMENTUM: panel},
    )

    assert context.universe is snapshot
    assert context.factor_panels[MOMENTUM] is panel


def test_a_copy_of_a_panel_frame_cannot_change_the_context() -> None:
    context = make_context()
    copy = context.factor_panels[MOMENTUM].frame
    copy.loc[0, "raw_value"] = 99.0

    assert context.factor_panels[MOMENTUM].frame.loc[0, "raw_value"] != 99.0


def test_contexts_are_compared_by_identity() -> None:
    first = make_context()
    second = make_context()

    assert first is not second
    assert first != second


@pytest.mark.parametrize(
    "as_of_date",
    [
        "2024-13-05",
        "2024-01-05T16:00",
        "2024-01-05T00:00",
        "2024-01-05 00:00:00",
        "05/01/2024",
        "20240105",
        datetime(2024, 1, 5, 16, 0),
        datetime(2024, 1, 5),
        datetime(2024, 1, 5, tzinfo=UTC),
        pd.Timestamp("2024-01-05"),
        None,
        20240105,
    ],
)
def test_context_rejects_an_unusable_date(as_of_date: object) -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        make_context(as_of_date=as_of_date)

    assert "StrategyContext.as_of_date" in str(excinfo.value)


def test_context_explains_why_a_timestamp_is_not_a_date() -> None:
    """Even midnight is a timestamp: a decision date has no time of day."""
    with pytest.raises(StrategyContractError) as excinfo:
        make_context(as_of_date=datetime(2024, 1, 5))

    message = str(excinfo.value)
    assert "StrategyContext.as_of_date" in message
    assert "not the timestamp" in message


@pytest.mark.parametrize("universe", [["AAPL", "MSFT", "NVDA"], ("AAPL", "MSFT", "NVDA"), 42])
def test_context_rejects_a_universe_that_is_not_a_snapshot(universe: object) -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        StrategyContext(
            as_of_date=AS_OF,
            universe=universe,  # type: ignore[arg-type]
            factor_panels={MOMENTUM: make_panel()},
        )

    assert "UniverseSnapshot" in str(excinfo.value)


def test_context_rejects_a_universe_from_another_session() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        StrategyContext(
            as_of_date=AS_OF,
            universe=make_universe(as_of_date=OTHER_SESSION),
            factor_panels={MOMENTUM: make_panel()},
        )

    message = str(excinfo.value)
    assert "2024-01-08" in message
    assert "2024-01-05" in message


def test_context_rejects_a_panel_from_another_session() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        StrategyContext(
            as_of_date=AS_OF,
            universe=make_universe(),
            factor_panels={MOMENTUM: make_panel(as_of_date=OTHER_SESSION)},
        )

    message = str(excinfo.value)
    assert "momentum_12_1/1.0.0" in message
    assert "2024-01-08" in message


@pytest.mark.parametrize(
    "panel_symbols",
    [
        ("AAPL", "MSFT"),
        ("AAPL", "MSFT", "GOOGL"),
        ("MSFT", "AAPL", "NVDA"),
    ],
)
def test_context_rejects_a_panel_of_another_universe(panel_symbols: tuple[str, ...]) -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        StrategyContext(
            as_of_date=AS_OF,
            universe=make_universe(),
            factor_panels={MOMENTUM: make_panel(symbols=panel_symbols)},
        )

    assert "momentum_12_1/1.0.0" in str(excinfo.value)


def test_context_rejects_a_key_that_is_not_the_panels_identity() -> None:
    renamed = FactorRequirement(factor_name="momentum_12_1", factor_version="2.0.0")

    with pytest.raises(StrategyContractError) as excinfo:
        StrategyContext(
            as_of_date=AS_OF,
            universe=make_universe(),
            factor_panels={renamed: make_panel()},
        )

    message = str(excinfo.value)
    assert "momentum_12_1/2.0.0" in message
    assert "identity" in message


def test_context_rejects_a_key_that_is_not_a_requirement() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        StrategyContext(
            as_of_date=AS_OF,
            universe=make_universe(),
            factor_panels={"momentum_12_1": make_panel()},  # type: ignore[dict-item]
        )

    assert "FactorRequirement" in str(excinfo.value)


@pytest.mark.parametrize("factor_panels", [{}, [], [(MOMENTUM, make_panel())], [(MOMENTUM,)]])
def test_context_rejects_an_unusable_mapping(factor_panels: object) -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        make_context(factor_panels)  # type: ignore[arg-type]

    assert "factor_panels" in str(excinfo.value)


def test_context_rejects_a_bare_frame_instead_of_a_mapping() -> None:
    """A frame has no factor identity, so it can never stand in for the mapping."""
    with pytest.raises(StrategyContractError) as excinfo:
        StrategyContext(
            as_of_date=AS_OF,
            universe=make_universe(),
            factor_panels=make_panel().frame,  # type: ignore[arg-type]
        )

    assert "not a DataFrame" in str(excinfo.value)


def test_context_rejects_a_bare_frame_instead_of_a_panel() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        StrategyContext(
            as_of_date=AS_OF,
            universe=make_universe(),
            factor_panels={MOMENTUM: make_panel().frame},  # type: ignore[dict-item]
        )

    assert "never takes a raw DataFrame" in str(excinfo.value)


def test_context_collects_every_panel_problem() -> None:
    """One error names the bad date, the bad identity, the missing symbol and the frame."""
    renamed = FactorRequirement(factor_name="momentum_12_1", factor_version="9.9.9")

    with pytest.raises(StrategyContractError) as excinfo:
        StrategyContext(
            as_of_date=AS_OF,
            universe=make_universe(),
            factor_panels={
                renamed: make_panel(as_of_date=OTHER_SESSION, symbols=("AAPL", "MSFT")),
                VALUE: make_panel().frame,  # type: ignore[dict-item]
            },
        )

    problems = excinfo.value.problems
    assert len(problems) == 4
    assert any("dated 2024-01-08" in problem for problem in problems)
    assert any("identity" in problem for problem in problems)
    assert any("missing universe symbol" in problem for problem in problems)
    assert any("never takes a raw DataFrame" in problem for problem in problems)


# ---------------------------------------------------------------------------
# StrategyResult: what a strategy returns
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# StrategyResult: the authoritative spec it carries
# ---------------------------------------------------------------------------
def test_strategy_result_carries_its_authoritative_spec() -> None:
    """A result's identity is the spec it was produced under -- not a string pair."""
    result = make_result()

    assert result.strategy_spec is SPEC
    assert result.strategy_name == SPEC.name == STRATEGY_NAME
    assert result.strategy_version == SPEC.version == STRATEGY_VERSION
    assert result.factor_requirements == SPEC.required_factors == (MOMENTUM,)


def test_factor_requirements_is_the_specs_own_ordered_tuple() -> None:
    """A tuple, in the declared order, naming each factor *and* its version."""
    multi = StrategySpec(
        name="blend_momentum_value",
        version="1.0.0",
        required_factors=(VALUE, MOMENTUM),
    )
    result = StrategyResult(
        make_result_frame(strategy_name=multi.name),
        universe=make_universe(),
        strategy_spec=multi,
    )

    assert isinstance(result.factor_requirements, tuple)
    assert result.factor_requirements is multi.required_factors
    assert result.factor_requirements == (VALUE, MOMENTUM)
    assert [requirement.factor_version for requirement in result.factor_requirements] == [
        "2.0.0",
        "1.0.0",
    ]


def test_the_result_identity_is_read_only() -> None:
    result = make_result()

    with pytest.raises(FrozenInstanceError):
        result.strategy_name = "other_strategy"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.factor_requirements = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.strategy_spec = SPEC_ON_FACTOR_V2  # type: ignore[misc]

    assert result.strategy_name == STRATEGY_NAME
    assert result.strategy_spec is SPEC


@pytest.mark.parametrize(
    "strategy_spec",
    [None, "top_n_momentum", "1.0.0", {"name": STRATEGY_NAME}, (SPEC,), MOMENTUM],
)
def test_strategy_result_requires_a_real_spec(strategy_spec: object) -> None:
    """A bare name, a version string or a dict is not a declaration."""
    with pytest.raises(StrategyContractError) as excinfo:
        make_result(strategy_spec=strategy_spec)

    message = str(excinfo.value)
    assert "strategy_spec" in message
    assert "StrategySpec" in message


def test_strategy_result_has_exactly_one_identity_argument() -> None:
    """``strategy_spec`` is required, and the old string pair is gone for good."""
    with pytest.raises(TypeError):
        StrategyResult(make_result_frame(), universe=make_universe())  # type: ignore[call-arg]

    for option in ({"strategy_name": STRATEGY_NAME}, {"strategy_version": STRATEGY_VERSION}):
        with pytest.raises(TypeError):
            make_result(**option)


def test_strategy_result_rejects_a_frame_named_after_another_strategy() -> None:
    frame = make_result_frame(strategy_name="other_strategy")

    with pytest.raises(StrategyContractError) as excinfo:
        make_result(frame)

    message = str(excinfo.value)
    assert "frame identity disagrees with strategy_spec" in message
    assert "other_strategy" in message
    assert STRATEGY_NAME in message


def test_strategy_result_rejects_a_frame_carrying_another_version() -> None:
    frame = make_result_frame(strategy_version="2.0.0")

    with pytest.raises(StrategyContractError) as excinfo:
        make_result(frame)

    message = str(excinfo.value)
    assert "frame identity disagrees with strategy_spec" in message
    assert "2.0.0" in message
    assert STRATEGY_VERSION in message


def test_two_factor_versions_are_distinguishable_from_the_result() -> None:
    """Same strategy name and version, different factor version: still traceable.

    This is the gap the spec closes: the frame alone is byte-for-byte identical,
    so only the attached spec can say which factor versions produced it -- which
    is what a later portfolio, backtest ledger or cache key must key on.
    """
    from_v1 = StrategyResult(
        make_result_frame(),
        universe=make_universe(),
        strategy_spec=SPEC,
    )
    from_v2 = StrategyResult(
        make_result_frame(),
        universe=make_universe(),
        strategy_spec=SPEC_ON_FACTOR_V2,
    )

    assert from_v1.strategy_name == from_v2.strategy_name == STRATEGY_NAME
    assert from_v1.strategy_version == from_v2.strategy_version == STRATEGY_VERSION
    assert from_v1.factor_requirements != from_v2.factor_requirements
    assert from_v1.strategy_spec != from_v2.strategy_spec
    assert {from_v1.strategy_spec, from_v2.strategy_spec} == {SPEC, SPEC_ON_FACTOR_V2}
    assert [str(requirement) for requirement in from_v2.factor_requirements] == [
        "momentum_12_1/2.0.0"
    ]

    pd.testing.assert_frame_equal(from_v1.frame, from_v2.frame)


def test_the_dependencies_are_not_repeated_as_frame_columns() -> None:
    """One row per symbol stays one row per symbol, however many factors are read."""
    result = make_result()

    assert tuple(result.frame.columns) == STRATEGY_RESULT_COLUMNS
    assert "factor_name" not in result.frame.columns
    assert "factor_version" not in result.frame.columns


def test_strategy_result_round_trips_the_frame() -> None:
    frame = make_result_frame()
    universe = make_universe()
    result = StrategyResult(frame, universe=universe, strategy_spec=SPEC)

    assert result.as_of_date == AS_OF
    assert result.strategy_name == STRATEGY_NAME
    assert result.strategy_version == STRATEGY_VERSION
    assert result.universe is universe
    assert result.symbols == SYMBOLS
    assert len(result) == len(SYMBOLS)

    round_trip = result.frame
    assert tuple(round_trip.columns) == STRATEGY_RESULT_COLUMNS
    assert round_trip["symbol"].tolist() == list(SYMBOLS)
    assert round_trip["strategy_name"].tolist() == [STRATEGY_NAME] * len(SYMBOLS)
    assert round_trip["strategy_version"].tolist() == [STRATEGY_VERSION] * len(SYMBOLS)
    assert round_trip["as_of_date"].tolist() == [pd.Timestamp(AS_OF)] * len(SYMBOLS)
    assert round_trip["score"].tolist() == pytest.approx([0.9, 0.5, 0.1])
    assert round_trip["rank"].tolist() == [1, 2, 3]
    assert round_trip["is_selected"].tolist() == [True, False, False]


def test_strategy_result_normalizes_the_column_dtypes() -> None:
    result = make_result(make_result_frame(score=[0.9, None, 0.1], rank=[1, None, 3]))

    frame = result.frame
    assert frame["score"].dtype == "float64"
    assert str(frame["rank"].dtype) == "Int64"
    assert frame["is_selected"].dtype == "bool"
    assert str(frame["as_of_date"].dtype) == "datetime64[s]"
    assert pd.isna(frame["rank"].tolist()[1])
    assert pd.isna(frame["score"].tolist()[1])


def test_a_midnight_datetime_column_is_normalized_to_the_session() -> None:
    """A ``datetime64`` column is how pandas stores a date, so it is normalized."""
    frame = make_result_frame()
    assert str(frame["as_of_date"].dtype).startswith("datetime64")

    result = make_result(frame)

    assert result.as_of_date == AS_OF
    assert isinstance(result.as_of_date, date)
    assert not isinstance(result.as_of_date, datetime)
    assert result.frame["as_of_date"].dt.date.tolist() == [AS_OF] * len(SYMBOLS)


def test_strategy_result_rejects_a_timestamp_argument() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        make_result(as_of_date=datetime(2024, 1, 5))

    message = str(excinfo.value)
    assert "StrategyResult.as_of_date" in message
    assert "not the timestamp" in message


def test_strategy_result_rejects_a_timezone_aware_date_column() -> None:
    frame = make_result_frame()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.tz_localize("America/New_York")

    with pytest.raises(StrategyContractError) as excinfo:
        make_result(frame)

    assert "timezone" in str(excinfo.value)


def test_strategy_result_accepts_a_date_column_of_plain_dates() -> None:
    frame = make_result_frame()
    frame["as_of_date"] = pd.Series([AS_OF] * len(SYMBOLS), dtype="object")

    result = make_result(frame)

    assert result.as_of_date == AS_OF
    assert result.frame["as_of_date"].dt.date.tolist() == [AS_OF] * len(SYMBOLS)


def test_strategy_result_accepts_an_iso_string_cross_check() -> None:
    result = make_result(as_of_date="2024-01-05")

    assert result.as_of_date == AS_OF

    shuffled = make_result_frame()

    result = make_result(shuffled.loc[:, list(reversed(STRATEGY_RESULT_COLUMNS))])

    assert tuple(result.frame.columns) == STRATEGY_RESULT_COLUMNS


def test_strategy_result_returns_a_defensive_copy() -> None:
    frame = make_result_frame()
    result = StrategyResult(frame, universe=make_universe(), strategy_spec=SPEC)

    copy = result.frame
    copy.loc[0, "score"] = 123.0
    copy.loc[0, "symbol"] = "XXXX"
    frame.loc[0, "score"] = -1.0

    assert result.frame.loc[0, "score"] == 0.9
    assert result.frame.loc[0, "symbol"] == "AAPL"
    assert result.frame is not result.frame


def test_strategy_result_accepts_an_empty_selection() -> None:
    """How many symbols a portfolio needs is not this layer's question."""
    result = make_result(make_result_frame(is_selected=[False, False, False]))

    assert result.selected_symbols == ()
    assert result.frame["is_selected"].tolist() == [False, False, False]


def test_strategy_result_accepts_a_missing_score() -> None:
    """``NaN`` means "not scored", which is legal for an unselected symbol."""
    result = make_result(
        make_result_frame(score=[0.9, float("nan"), 0.1], rank=[1, 2, 3]),
    )

    scores = result.frame["score"].tolist()
    assert scores[0] == 0.9
    assert pd.isna(scores[1])
    assert scores[2] == 0.1
    assert result.selected_symbols == ("AAPL",)


def test_strategy_results_are_compared_by_identity() -> None:
    first = make_result()
    second = make_result()

    assert first is not second
    assert first != second


def test_strategy_result_rejects_missing_symbols() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        make_result(make_result_frame(SYMBOLS[:-1]))

    message = str(excinfo.value)
    assert "NVDA" in message
    assert "cover" in message


def test_strategy_result_rejects_symbols_outside_the_universe() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        make_result(make_result_frame(("AAPL", "MSFT", "GOOGL")))

    message = str(excinfo.value)
    assert "GOOGL" in message
    assert "outside the universe" in message


def test_strategy_result_rejects_a_repeated_symbol() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        make_result(make_result_frame(("AAPL", "AAPL", "NVDA")))

    assert "more than once" in str(excinfo.value)


def test_strategy_result_rejects_a_wrong_row_order() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        make_result(make_result_frame(("MSFT", "AAPL", "NVDA")))

    message = str(excinfo.value)
    assert "order" in message
    assert "row 0" in message


@pytest.mark.parametrize("as_of_date", [OTHER_SESSION, "2024-01-08"])
def test_strategy_result_rejects_a_contradicting_date_argument(as_of_date: object) -> None:
    """The frame is the date; an argument that disagrees is a bug."""
    with pytest.raises(StrategyContractError) as excinfo:
        make_result(as_of_date=as_of_date)

    assert "was built for" in str(excinfo.value)


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("as_of_date", [AS_OF, OTHER_SESSION, AS_OF]),
        ("as_of_date", [AS_OF, None, AS_OF]),
        ("as_of_date", [AS_OF, "not-a-date", AS_OF]),
        ("as_of_date", [AS_OF, "2024-01-05T16:00", AS_OF]),
        ("as_of_date", [datetime(2024, 1, 5, 16, 0)] * 3),
        ("as_of_date", [datetime(2024, 1, 5)] * 3),
        ("strategy_name", [STRATEGY_NAME, "other_strategy", STRATEGY_NAME]),
        ("strategy_name", [STRATEGY_NAME, "Top N", STRATEGY_NAME]),
        ("strategy_version", [STRATEGY_VERSION, "2.0.0", STRATEGY_VERSION]),
        ("strategy_version", [STRATEGY_VERSION, "", STRATEGY_VERSION]),
    ],
)
def test_strategy_result_rejects_a_contradicting_identity(
    column: str,
    values: list[object],
) -> None:
    frame = make_result_frame()
    frame[column] = pd.Series(values, dtype="object")

    with pytest.raises(StrategyContractError) as excinfo:
        make_result(frame)

    assert column in str(excinfo.value)


def test_strategy_result_rejects_a_non_frame() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        make_result([{"symbol": "AAPL"}])

    assert "DataFrame" in str(excinfo.value)


def test_strategy_result_rejects_an_empty_frame() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        make_result(make_result_frame().iloc[0:0])

    assert "not be empty" in str(excinfo.value)


def test_strategy_result_rejects_a_universe_that_is_not_a_snapshot() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        make_result(universe=list(SYMBOLS))

    assert "UniverseSnapshot" in str(excinfo.value)


@pytest.mark.parametrize("column", FORBIDDEN_RESULT_COLUMNS)
def test_strategy_result_rejects_a_portfolio_or_trading_column(column: str) -> None:
    """A weight, a quantity, a side or a fill belongs to a later layer."""
    frame = make_result_frame()
    frame[column] = 0.0

    with pytest.raises(StrategyContractError) as excinfo:
        make_result(frame)

    message = str(excinfo.value)
    assert column in message
    assert "later layer" in message


def test_strategy_result_rejects_missing_or_unexpected_columns() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        make_result(make_result_frame().drop(columns=["rank"]))
    assert "missing column(s) ['rank']" in str(excinfo.value)

    extra = make_result_frame()
    extra["zscore"] = 0.0  # transforms belong to a later module
    with pytest.raises(StrategyContractError) as excinfo:
        make_result(extra)
    assert "zscore" in str(excinfo.value)


@pytest.mark.parametrize(
    "score",
    [
        ["0.9", "0.5", "0.1"],
        [True, False, True],
        [0.9, object(), 0.1],
    ],
)
def test_strategy_result_rejects_a_non_numeric_score(score: object) -> None:
    frame = make_result_frame(is_selected=[False, False, False])
    frame["score"] = pd.Series(score, dtype="object")

    with pytest.raises(StrategyContractError) as excinfo:
        make_result(frame)

    assert "score must be numeric" in str(excinfo.value)


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_strategy_result_rejects_an_infinite_score(value: float) -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        make_result(
            make_result_frame(score=[value, 0.5, 0.1], is_selected=[False, False, False]),
        )

    assert "finite" in str(excinfo.value)


@pytest.mark.parametrize(
    "rank",
    [
        [0, 2, 3],
        [-1, 2, 3],
        [1.5, 2, 3],
        [1, 1, 3],
        ["1", "2", "3"],
        [float("inf"), 2, 3],
    ],
)
def test_strategy_result_rejects_an_invalid_rank(rank: object) -> None:
    frame = make_result_frame(rank=rank, is_selected=[False, False, False])

    with pytest.raises(StrategyContractError) as excinfo:
        make_result(frame)

    assert "rank" in str(excinfo.value)


@pytest.mark.parametrize(
    "selected",
    [
        [1, 0, 0],
        ["yes", "no", "no"],
        [None, False, False],
        [1.0, 0.0, 0.0],
    ],
)
def test_strategy_result_rejects_a_non_boolean_is_selected(selected: object) -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        make_result(make_result_frame(is_selected=selected))

    assert "is_selected" in str(excinfo.value)


def test_strategy_result_rejects_a_missing_selection_flag() -> None:
    flags = pd.array([True, pd.NA, False], dtype="boolean")

    with pytest.raises(StrategyContractError) as excinfo:
        make_result(make_result_frame(is_selected=flags))

    assert "is_selected" in str(excinfo.value)


def test_strategy_result_rejects_a_selection_without_a_score() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        make_result(make_result_frame(score=[float("nan"), 0.5, 0.1]))

    assert "no finite score" in str(excinfo.value)


def test_strategy_result_rejects_a_selection_without_a_rank() -> None:
    with pytest.raises(StrategyContractError) as excinfo:
        make_result(make_result_frame(rank=[None, 2, 3]))

    assert "no rank" in str(excinfo.value)


def test_strategy_result_collects_every_problem() -> None:
    """One error names the symbols, the rank and the score that are wrong."""
    frame = make_result_frame()
    frame.loc[1, "symbol"] = "GOOGL"
    frame.loc[0, "rank"] = 0
    frame.loc[0, "score"] = float("inf")

    with pytest.raises(StrategyContractError) as excinfo:
        make_result(frame)

    problems = excinfo.value.problems
    assert len(problems) == 5
    joined = "\n".join(problems)
    assert "outside the universe" in joined
    assert "missing ['MSFT']" in joined
    assert "rank starts at 1" in joined
    assert "score must be finite" in joined
    assert "no finite score" in joined


# ---------------------------------------------------------------------------
# Package surface and architectural boundary
# ---------------------------------------------------------------------------
def test_the_contract_is_exported_from_the_package_root() -> None:
    expected = {
        "FORBIDDEN_RESULT_COLUMNS",
        "STRATEGY_NAME_PATTERN",
        "STRATEGY_RESULT_COLUMNS",
        "FactorRequirement",
        "Strategy",
        "StrategyContext",
        "StrategyContractError",
        "StrategyError",
        "StrategyResult",
        "StrategySpec",
    }

    assert expected <= set(strategy_sys.__all__)
    assert strategy_sys.StrategySpec is StrategySpec
    assert strategy_sys.StrategyContext is StrategyContext
    assert strategy_sys.StrategyResult is StrategyResult
    assert strategy_sys.FactorRequirement is FactorRequirement
    assert strategy_sys.Strategy is Strategy
    assert strategy_sys.StrategyContractError is StrategyContractError
    assert strategy_sys.StrategyError is StrategyError
    assert contracts_module.__all__ == [
        "FORBIDDEN_RESULT_COLUMNS",
        "STRATEGY_NAME_PATTERN",
        "STRATEGY_RESULT_COLUMNS",
        "FactorRequirement",
        "Strategy",
        "StrategyContext",
        "StrategyResult",
        "StrategySpec",
    ]


def test_strategy_errors_are_their_own_hierarchy() -> None:
    """A strategy violation is neither a factor nor a data-pipeline error."""
    assert issubclass(StrategyContractError, StrategyError)
    assert not issubclass(StrategyError, FactorError)
    assert not issubclass(StrategyError, DataPipelineError)
    assert not issubclass(StrategyError, ValueError)
    assert not issubclass(StrategyError, KeyError)


def test_strategy_contract_errors_carry_their_problems() -> None:
    error = StrategyContractError("invalid thing", ["first problem", "second problem"])

    assert error.problems == ["first problem", "second problem"]
    assert "first problem" in str(error)
    assert "second problem" in str(error)
    assert str(error).startswith("invalid thing\n  - ")
    assert StrategyContractError("plain").problems == []


#: Modules the contract may import: the standard library it needs, pandas, the
#: factor layer's validated objects, and the layer's own runner -- nothing else.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "abc",
        "collections.abc",
        "dataclasses",
        "datetime",
        "re",
        "types",
        "typing",
        "pandas",
        "factor_sys.contracts",
        "factor_sys.universe",
        "strategy_sys.contracts",
        "strategy_sys.errors",
        "strategy_sys.registry",
        "strategy_sys.runner",
    }
)

#: Text that must never appear in the contract modules: storage, network, market
#: data, calendars, factor plumbing or portfolio construction.
FORBIDDEN_TEXT: tuple[str, ...] = (
    "from data_sys",
    "import data_sys",
    "factor_sys.runner",
    "factor_sys.factor_store",
    "factor_sys.storage",
    "factor_sys.registry",
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
)


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


def contract_modules() -> list[Path]:
    """The three modules that make up the strategy contract layer."""
    package = Path(strategy_sys.__file__).parent
    return [package / "__init__.py", package / "contracts.py", package / "errors.py"]


def test_the_contract_imports_only_its_own_layers() -> None:
    """No store, no storage, no network, no calendar: pandas and the contracts."""
    for module in contract_modules():
        imported = imported_modules(module)
        assert imported <= ALLOWED_IMPORTS, (module.name, sorted(imported - ALLOWED_IMPORTS))


def contract_code(path: Path) -> str:
    """``path``'s source with every docstring blanked out.

    Prose is allowed to *name* what the layer does not use -- the errors module
    documents which store errors travel back untouched -- while the code itself
    must never import or call any of it.
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    documented = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, documented) and ast.get_docstring(node, clean=False) is not None:
            docstring = node.body[0]
            for index in range(docstring.lineno - 1, docstring.end_lineno):
                lines[index] = ""
    return "".join(lines)


def test_the_contract_modules_never_name_a_forbidden_dependency() -> None:
    for module in contract_modules():
        code = contract_code(module)
        offenders = [text for text in FORBIDDEN_TEXT if text in code]
        assert offenders == [], (module.name, offenders)


def test_the_contract_never_opens_or_prints_anything() -> None:
    """The layer calls no builtin that reads, writes or prints."""
    for module in contract_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        called = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        assert called & {"open", "print", "input", "eval", "exec", "compile"} == set(), module.name


def test_the_contract_stays_deliberately_small() -> None:
    """Types, validation and the interface only: no rule and no portfolio."""
    assert {record.name for record in fields(FactorRequirement)} == {
        "factor_name",
        "factor_version",
    }
    assert {record.name for record in fields(StrategyResult)} == {
        "as_of_date",
        "universe",
        "strategy_spec",
        "_rows",
    }
    # The identity and dependency views are properties of the spec, not extra
    # stored fields that could disagree with it.
    assert [
        name
        for name in ("strategy_name", "strategy_version", "factor_requirements")
        if isinstance(getattr(StrategyResult, name), property)
    ] == ["strategy_name", "strategy_version", "factor_requirements"]

    for absent in (
        "TopNByFactorStrategy",
        "PortfolioConstructor",
        "load_panels",
        "run_strategy",
    ):
        assert not hasattr(strategy_sys, absent)

    assert "weight" not in STRATEGY_RESULT_COLUMNS
    assert "quantity" not in STRATEGY_RESULT_COLUMNS
