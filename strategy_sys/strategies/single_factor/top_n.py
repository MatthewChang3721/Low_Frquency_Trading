"""Select the ``N`` highest factor values of one cross-section -- and nothing else.

Where this module sits
----------------------
The strategy layer is a pure function of what it is given::

    FactorPanel  ->  Strategy  ->  StrategyResult  ->  (portfolio, later)

:class:`~strategy_sys.contracts.Strategy` defines the interface, and this module
implements one rule of it.  A strategy reads a validated
:class:`~strategy_sys.contracts.StrategyContext`, scores the symbols it can score
and returns a validated :class:`~strategy_sys.contracts.StrategyResult`.  It has
no opinion about weights, cash, position sizes, orders, fills or returns, so it
never needs a market price, a trading calendar or a file.

Why the rule is generic
-----------------------
:class:`TopNByFactorStrategy` is *not* a momentum strategy.  It is given one
exact :class:`~strategy_sys.contracts.FactorRequirement` and sorts by that
factor's ``raw_value``, so the same class is a momentum rule, a value rule or a
quality rule depending on the panel the runner hands over::

    TopNByFactorStrategy(FactorRequirement("momentum_12_1", "1.0.0"), top_n=10)
    # spec: top_10_by_momentum_12_1 / 1.0.0

A subclass that only fixes the factor (``MomentumTopNStrategy``) would add a name
and no behaviour, so it is deliberately not written here.

The rule, exactly
-----------------
1. The panel is looked up in ``context.factor_panels`` by *the* requirement the
   strategy declared -- name and version both.  A missing panel is an error, not
   an empty result.
2. The result carries one row per ``context.universe.symbols`` entry, **in that
   order**.  Ordering the cross-section is the job of the ``rank`` column: the
   frame is never re-sorted, so row ``i`` always describes the same symbol for
   every strategy and every date.
3. A symbol competes for a place only when its ``is_eligible`` flag is ``True``
   *and* its ``raw_value`` is a finite number (``NaN``, ``+inf`` and ``-inf`` are
   not values, they are missing answers).
4. Rankable symbols get ``score = raw_value``, are ordered by score descending
   and, on an exactly equal score, by symbol ascending -- the only tie-break,
   which makes the ranking total and reproducible.  Ranks are then ``1, 2, 3,
   ...`` with no gaps and no repeats; the first ``top_n`` are selected.
5. Every other symbol keeps ``score = NaN``, ``rank = pd.NA`` and
   ``is_selected = False``: "not scored" is a fact the contract types, and it is
   never the same statement as "scored low".
6. The result carries this strategy's whole ``spec``, so
   ``result.factor_requirements`` names the factor **and** its exact version even
   when two strategies share a strategy name and version.  The dependency is
   metadata on the result, deliberately never a column repeated per symbol.

``top_n`` larger than the number of rankable symbols selects them all, and a
cross-section with nothing rankable returns a legal empty selection.  Neither is
an error: how many names a portfolio needs is the portfolio layer's decision.

What this module must never do
------------------------------
Only the standard library, :mod:`pandas`, :mod:`strategy_sys.contracts` and
:mod:`strategy_sys.errors` are imported.  No storage, no factor runner, no market
data, no calendar, no network, no file.  The context, its panels and every frame
are read, never written, and ``raw_value`` is used as it is: no winsorizing, no
``zscore``, no rank transform -- a transform belongs to a factor, not to a
selection rule.

Versioning
----------
``spec.version`` labels *this behaviour*.  Changing the sort direction, the
eligibility rule, the tie-break or the meaning of ``score`` changes what the
strategy answers, so it requires a new version; the name already carries
``top_n`` and the factor, so a different ``top_n`` is a different strategy
identity on its own.

Examples
--------
>>> from strategy_sys import FactorRequirement
>>> strategy = TopNByFactorStrategy(
...     factor_requirement=FactorRequirement("momentum_12_1", "1.0.0"),
...     top_n=10,
... )
>>> strategy.spec.name
'top_10_by_momentum_12_1'
>>> [str(requirement) for requirement in strategy.spec.required_factors]
['momentum_12_1/1.0.0']
"""

from __future__ import annotations

from numbers import Integral

import pandas as pd

from strategy_sys.contracts import (
    FactorRequirement,
    Strategy,
    StrategyContext,
    StrategyResult,
    StrategySpec,
)
from strategy_sys.errors import StrategyContractError

#: Default version of this rule (see "Versioning" above).
DEFAULT_VERSION: str = "1.0.0"

_VALUE_COLUMN = "raw_value"
_ELIGIBLE_COLUMN = "is_eligible"


class TopNByFactorStrategy(Strategy):
    """Select the ``top_n`` symbols with the highest ``raw_value`` of one factor.

    Parameters
    ----------
    factor_requirement:
        The exact factor identity to read: ``factor_name`` *and*
        ``factor_version``.  It becomes the strategy's only required factor, so
        the panel this strategy consumes is part of its identity.
    top_n:
        How many rankable symbols to select: a positive integer.  ``True``,
        ``1.0`` and ``"10"`` are refused instead of being converted, because a
        selection size that was silently changed is a different strategy.
    version:
        The strategy version, stored in :attr:`spec` and in every result.

    Raises
    ------
    strategy_sys.errors.StrategyContractError
        If ``factor_requirement`` is not a
        :class:`~strategy_sys.contracts.FactorRequirement`, if ``top_n`` is not a
        positive integer, or if ``version`` is not a non-empty string.  All
        problems are reported at once.

    Notes
    -----
    The instance is read-only: :attr:`spec`, :attr:`factor_requirement`,
    :attr:`top_n` and :attr:`version` have no setters, so a strategy handed to a
    runner cannot have its rule changed underneath it.  Building one costs
    nothing and never touches data, which makes it safe to construct per run.

    Examples
    --------
    >>> from strategy_sys import FactorRequirement
    >>> strategy = TopNByFactorStrategy(FactorRequirement("book_to_price", "2.0.0"), 5)
    >>> strategy.spec.name
    'top_5_by_book_to_price'
    >>> strategy.top_n
    5
    """

    __slots__ = ("_factor_requirement", "_spec", "_top_n", "_version")

    def __init__(
        self,
        factor_requirement: FactorRequirement,
        top_n: int,
        *,
        version: str = DEFAULT_VERSION,
    ) -> None:
        """Validate the rule's parameters and freeze its identity."""
        problems: list[str] = []

        if not isinstance(factor_requirement, FactorRequirement):
            problems.append(
                "factor_requirement must be a FactorRequirement -- the exact "
                "(factor_name, factor_version) identity -- got "
                f"{type(factor_requirement).__name__}: {factor_requirement!r}; a bare name or "
                "a tuple carries no version, so two versions of one factor could not be told "
                "apart"
            )
            factor_requirement = None  # type: ignore[assignment]

        if isinstance(top_n, bool) or not isinstance(top_n, Integral):
            problems.append(
                f"top_n must be a positive integer, got {type(top_n).__name__}: {top_n!r}; a "
                "bool, a float or a string is refused rather than converted, because 'top 10' "
                "has to mean exactly the integer 10"
            )
            top_n = 0  # type: ignore[assignment]
        elif top_n <= 0:
            problems.append(f"top_n must be greater than zero, got {top_n!r}")
            top_n = 0  # type: ignore[assignment]

        if not isinstance(version, str) or not version.strip():
            problems.append(f"version must be a non-empty string, got {version!r}")

        if problems:
            raise StrategyContractError("invalid TopNByFactorStrategy", problems)

        size = int(top_n)
        spec = StrategySpec(
            name=f"top_{size}_by_{factor_requirement.factor_name}",
            version=version,
            required_factors=(factor_requirement,),
        )

        self._factor_requirement = factor_requirement
        self._top_n = size
        self._version = version
        self._spec = spec

    @property
    def spec(self) -> StrategySpec:
        """The rule's identity: ``top_<n>_by_<factor>``, its version, its factor."""
        return self._spec

    @property
    def factor_requirement(self) -> FactorRequirement:
        """The one factor identity this strategy reads."""
        return self._factor_requirement

    @property
    def top_n(self) -> int:
        """How many rankable symbols are selected per cross-section."""
        return self._top_n

    @property
    def version(self) -> str:
        """The rule's version, mirroring ``spec.version``."""
        return self._version

    def compute(self, context: StrategyContext) -> StrategyResult:
        """Rank ``context``'s universe by the factor and select the top ``top_n``.

        Parameters
        ----------
        context:
            The point-in-time input: one decision date, one universe and the
            panels the runner produced.  It is read only -- nothing inside it,
            and no frame it hands out, is modified -- so the same context always
            yields an equal result.

        Returns
        -------
        StrategyResult
            One row per ``context.universe.symbols`` symbol, in that order, with
            ``score``, ``rank`` and ``is_selected`` as described in the module
            docstring, dated ``context.as_of_date`` and carrying :attr:`spec`
            itself -- so ``result.factor_requirements`` is exactly the factor
            name and version this rule read, not a per-row column.

        Raises
        ------
        strategy_sys.errors.StrategyContractError
            If ``context`` is not a
            :class:`~strategy_sys.contracts.StrategyContext`, or if it does not
            carry a panel for :attr:`factor_requirement`.  The error names the
            missing factor and the factors the context does carry; no partial
            result is returned.

        Examples
        --------
        >>> from factor_sys import FactorPanel, UniverseSnapshot
        >>> from strategy_sys import FactorRequirement, StrategyContext
        >>> universe = UniverseSnapshot(
        ...     as_of_date="2024-01-05",
        ...     symbols=("MSFT", "AAPL", "NVDA", "TSLA"),
        ...     source="config/universe_seed.csv",
        ... )
        >>> panel = FactorPanel(
        ...     pd.DataFrame(
        ...         {
        ...             "as_of_date": [pd.Timestamp("2024-01-05")] * 4,
        ...             "symbol": ["MSFT", "AAPL", "NVDA", "TSLA"],
        ...             "factor_name": ["momentum_12_1"] * 4,
        ...             "factor_version": ["1.0.0"] * 4,
        ...             "raw_value": [0.20, 0.20, 0.10, float("nan")],
        ...             "is_eligible": [True, True, True, False],
        ...         }
        ...     ),
        ...     universe=universe.symbols,
        ...     as_of_date=universe.as_of_date,
        ... )
        >>> context = StrategyContext(
        ...     as_of_date="2024-01-05",
        ...     universe=universe,
        ...     factor_panels={FactorRequirement("momentum_12_1", "1.0.0"): panel},
        ... )
        >>> strategy = TopNByFactorStrategy(FactorRequirement("momentum_12_1", "1.0.0"), 2)
        >>> result = strategy.compute(context)
        >>> result.selected_symbols
        ('MSFT', 'AAPL')
        >>> list(result.frame["symbol"])
        ['MSFT', 'AAPL', 'NVDA', 'TSLA']
        >>> result.frame["rank"].tolist()
        [2, 1, 3, <NA>]
        >>> result.frame["is_selected"].tolist()
        [True, True, False, False]
        """
        if not isinstance(context, StrategyContext):
            raise StrategyContractError(
                "TopNByFactorStrategy.compute needs a StrategyContext, got "
                f"{type(context).__name__}: a strategy reads only the validated context it is "
                "handed, never a frame or a bare panel"
            )

        spec = self._spec
        requirement = self._factor_requirement
        try:
            panel = context.factor_panels[requirement]
        except KeyError:
            carried = [str(key) for key in context.factor_panels]
            raise StrategyContractError(
                f"{spec.name} needs the factor {requirement}, which this StrategyContext "
                "does not carry",
                [
                    "the context carries "
                    + (", ".join(carried) if carried else "no factors at all"),
                    "a strategy looks its requirements up by exact identity and never "
                    "substitutes or invents a panel, so no partial result is returned",
                ],
            ) from None

        symbols = context.universe.symbols
        # ``StrategyContext`` already proved that this panel covers ``symbols`` in
        # this exact order, so the panel's rows line up with the universe row for
        # row; reading them positionally is what keeps the result in universe
        # order instead of in score order.
        frame = panel.frame
        values = frame[_VALUE_COLUMN].astype("float64").tolist()
        eligible = frame[_ELIGIBLE_COLUMN].astype(bool).tolist()

        scores = [float("nan")] * len(symbols)
        ranks: list[int | None] = [None] * len(symbols)
        selected = [False] * len(symbols)

        rankable = [
            position
            for position in range(len(symbols))
            if bool(eligible[position])
            and not pd.isna(values[position])
            and abs(values[position]) != float("inf")
        ]
        # Descending score, then ascending symbol: the score order is the rule and
        # the symbol is the only tie-break, so the ranking is total (symbols are
        # unique within a universe) and therefore reproducible across runs.
        rankable.sort(key=lambda position: (-values[position], symbols[position]))

        for rank, position in enumerate(rankable, start=1):
            scores[position] = values[position]
            ranks[position] = rank
            selected[position] = rank <= self._top_n

        rows = pd.DataFrame(
            {
                "as_of_date": [pd.Timestamp(context.as_of_date)] * len(symbols),
                "strategy_name": [spec.name] * len(symbols),
                "strategy_version": [spec.version] * len(symbols),
                "symbol": list(symbols),
                "score": pd.Series(scores, dtype="float64"),
                "rank": pd.Series(ranks, dtype="Int64"),
                "is_selected": pd.Series(selected, dtype="bool"),
            }
        )
        return StrategyResult(
            rows,
            universe=context.universe,
            strategy_spec=spec,
            as_of_date=context.as_of_date,
        )


__all__ = ["DEFAULT_VERSION", "TopNByFactorStrategy"]
