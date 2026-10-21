"""The 12-1 momentum factor: the project's first real formula.

What it computes
----------------
For a decision session ``t`` and one symbol::

    momentum_12_1(t) = adjusted_close(t - 21) / adjusted_close(t - 252) - 1

* ``adjusted_close(t - 252)`` is the start of the holding period: roughly twelve
  months of (split- and dividend-adjusted) price before ``t``.
* ``adjusted_close(t - 21)`` is the end of it: roughly one month before ``t``.
* The most recent 21 sessions are skipped **on purpose**.  The last month of a
  stock's price is dominated by short-term reversal, which is a different effect
  with a different sign; classic 12-1 momentum measures the year *before* that
  month.
* ``t`` itself never enters the return.  ``as_of_date`` only decides how much
  information is visible and which cross-section the values belong to, so the
  number is the same whether the panel is built on ``t`` or on a later session.

``adjusted_close`` is the only field the formula reads: splits and dividends
must be removed *before* a twelve-month return is taken, and the raw ``close``,
``open`` or a vendor's ``daily_return`` are not substitutes.

Why the anchors are taken by position, from the end
---------------------------------------------------
:class:`~factor_sys.runner.FactorRunner` asks a :class:`MarketDataReader` for
``lookback_sessions + 1 = 253`` sessions ending on ``t``, so a complete history
is exactly ``[t - 252, ..., t]``: 253 observations, with ``t - 252`` first and
``t - 21`` 22nd from the end.  The factor therefore anchors on positions
``-253`` and ``-22`` of the symbol's own date-sorted history and never calls a
trading calendar itself.  This module has no idea what a session is, which
exchange it trades on, or which files hold the bars -- that is the runner's job,
and keeping it there is what makes a factor testable from one in-memory frame.

A missing session inside the runner's window leaves only 252 observations, so
the "at least 253" rule below turns the symbol ineligible instead of silently
comparing two prices that are not one year/one month apart.  Missing sessions are
never filled in, never interpolated and never replaced by the nearest available
price: a factor that guesses is a factor whose backtest cannot be reproduced.

Eligibility
-----------
Every symbol of ``context.universe`` gets exactly one row, no matter how little
data it has:

``raw_value = adjusted_close(t - 21) / adjusted_close(t - 252) - 1``,
``is_eligible = True``
    the symbol has a bar on ``t``, at least 253 observations, and both anchors
    are numeric, finite and strictly positive.

``raw_value = NaN``, ``is_eligible = False``
    anything else: no bars at all, fewer than 253 observations, no bar dated
    ``t`` (a delisted or stale history), or an anchor that is missing,
    non-numeric, infinite, zero or negative.  A ratio built on zero or a
    negative price is not a return, so the row is reported as ineligible rather
    than invented.

One bad symbol never fails the run and is never dropped: the panel still covers
the whole universe, and downstream code decides what an ineligible row means.

Contract errors
---------------
If ``context.bars`` has no ``"adjusted_close"`` column the factor raises
:class:`~factor_sys.errors.FactorContractError`.  That is a caller error -- a
reader that ignored ``FactorSpec.required_fields`` -- and it must be loud, not
answered with a panel of 253 ``NaN`` rows.

Purity
------
The factor is pure computation over the context: it reads no file, writes no
file, opens no connection, reads no clock, knows no path and mutates neither the
context nor its bars.  The same context always produces the same panel.

Examples
--------
>>> Momentum12_1().spec.name
'momentum_12_1'
>>> Momentum12_1().spec.required_fields
('adjusted_close',)
"""

from __future__ import annotations

import math
from datetime import date
from numbers import Real
from typing import Any

import pandas as pd

from factor_sys.contracts import Factor, FactorContext, FactorPanel, FactorSpec
from factor_sys.errors import FactorContractError

#: The only price field a twelve-month return may be taken from.
PRICE_FIELD: str = "adjusted_close"

#: Sessions between the end anchor and the decision date (the skipped month).
SKIP_SESSIONS: int = 21

#: Sessions between the start anchor and the decision date (the skipped year).
LOOKBACK_SESSIONS: int = 252

#: Observations a complete window holds: the 252 past sessions and ``t`` itself.
REQUIRED_OBSERVATIONS: int = LOOKBACK_SESSIONS + 1

#: Declared once, next to the formula, and never rebuilt: a factor's identity and
#: its input requirements must not drift between runs.
SPEC = FactorSpec(
    name="momentum_12_1",
    version="1.0.0",
    required_fields=(PRICE_FIELD,),
    lookback_sessions=LOOKBACK_SESSIONS,
)


class Momentum12_1(Factor):
    """Twelve-month return, skipping the most recent month.

    ``adjusted_close(t - 21) / adjusted_close(t - 252) - 1`` -- see the module
    docstring for the anchors, the skip rule and the eligibility rules.
    """

    spec = SPEC

    def compute(self, context: FactorContext) -> FactorPanel:
        """Return this factor's one-row-per-symbol panel for ``context``.

        Raises
        ------
        factor_sys.errors.FactorContractError
            If ``context.bars`` does not carry the declared ``"adjusted_close"``
            field.  Insufficient or unusable *data* is not an error: it yields an
            ineligible row.
        """
        bars = context.bars
        if PRICE_FIELD not in bars.columns:
            raise FactorContractError(
                f"{type(self).__name__} needs the {PRICE_FIELD!r} column in "
                f"context.bars (FactorSpec.required_fields={self.spec.required_fields!r}) "
                f"but the frame has columns {list(bars.columns)!r}: a reader must return "
                "every field the spec declares, and the unadjusted 'close' is not a "
                "substitute for it"
            )

        histories = _histories(bars)
        symbols = context.universe

        values: list[float] = []
        eligible: list[bool] = []
        for symbol in symbols:
            value = _momentum_value(histories.get(symbol), context.as_of_date)
            values.append(math.nan if value is None else value)
            eligible.append(value is not None)

        count = len(symbols)
        frame = pd.DataFrame(
            {
                "as_of_date": [pd.Timestamp(context.as_of_date)] * count,
                "symbol": list(symbols),
                "factor_name": [self.spec.name] * count,
                "factor_version": [self.spec.version] * count,
                "raw_value": pd.Series(values, dtype="float64"),
                "is_eligible": pd.Series(eligible, dtype="bool"),
            }
        )
        return FactorPanel(frame, universe=symbols, as_of_date=context.as_of_date)


# ---------------------------------------------------------------------------
# Formula internals
# ---------------------------------------------------------------------------
def _histories(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split ``bars`` into one frame per symbol, without touching the input.

    ``groupby`` hands back a view whose row order follows the input, so each
    history is sorted by ``date`` when it is read (:func:`_momentum_value`).  The
    input frame is only ever read, never reassigned, so the caller's context stays
    exactly as it was.
    """
    return {symbol: group for symbol, group in bars.groupby("symbol", sort=False)}


def _momentum_value(history: pd.DataFrame | None, as_of_date: date) -> float | None:
    """The raw 12-1 return for one symbol, or ``None`` when it is ineligible."""
    if history is None or history.empty:
        return None

    ordered = history.sort_values("date", kind="stable")
    if pd.Timestamp(ordered["date"].iloc[-1]).date() != as_of_date:
        # No bar on the decision day: a stale or delisted history cannot be
        # priced against today's cross-section.
        return None
    if len(ordered) < REQUIRED_OBSERVATIONS:
        # A short window, or a window with a missing session, means the anchors
        # are not twelve months/one month apart.  Do not guess.
        return None

    prices = ordered[PRICE_FIELD]
    # ``-REQUIRED_OBSERVATIONS`` is the first row of the complete window (t-252),
    # ``-SKIP_SESSIONS - 1`` is the row twenty-two from the end (t-21).
    start = _positive_price(prices.iloc[-REQUIRED_OBSERVATIONS])
    end = _positive_price(prices.iloc[-SKIP_SESSIONS - 1])
    if start is None or end is None:
        return None

    value = end / start - 1.0
    return value if math.isfinite(value) else None


def _positive_price(value: Any) -> float | None:
    """``value`` as a usable price, or ``None`` if it is not one.

    A price has to be a real number (``bool`` is an ``int`` in Python, so it is
    rejected explicitly), finite and strictly positive.  Strings -- even numeric
    ones -- are refused: the storage layer writes prices as numbers, and coercing
    text here would hide a broken read.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    price = float(value)
    if not math.isfinite(price) or price <= 0.0:
        return None
    return price


__all__ = ["Momentum12_1"]
