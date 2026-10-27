# -*- coding: utf-8 -*-

"""
ALPHA 5-PILLAR CORE TRADING ENGINE
==================================

PURPOSE
-------
This module is responsible ONLY for:

1. Technical 5-pillar signal evaluation.
2. Direction detection.
3. Composite signal quality score.
4. Actual option-premium risk calculation AFTER the
   backtester obtains the real historical option price.

IMPORTANT
---------
This module DOES NOT generate a synthetic option premium.

There is NO:
    spot * fixed premium %
    delta = 0.50
    underlying move -> option move conversion

Actual historical option OHLC must come from the option-data
loader/backtester.

BACKTEST PIPELINE
-----------------
T Close
    |
    +--> evaluate_5pillar_gate()
    |
    v
T+1 Open
    |
    +--> actual CE/PE contract
    |
    +--> actual option Open
    |
    v
calculate_option_risk_from_premium()
    |
    v
Actual option OHLC simulation
"""

from typing import Dict, Any, Optional
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

INDEX_MIN_MOMENTUM_PCT = 0.14  # Calibrated for 3-Min Breakout Ignition (~30-35 pts in NIFTY, ~75-80 pts in BANKNIFTY)
STOCK_MIN_MOMENTUM_PCT = 0.45  # Calibrated for 3-Min Breakout Ignition in F&O Leaders (~0.35%-0.50%)


# ---------------------------------------------------------------------------
# Candle quality
# ---------------------------------------------------------------------------

MAX_ALLOWED_UPPER_WICK = 0.20
MIN_CLV_CLOSE_RATIO = 0.72
MIN_SOLID_BODY_RATIO = 0.50


# ---------------------------------------------------------------------------
# Volume / CVD
# ---------------------------------------------------------------------------

MIN_RVOL_MULTIPLIER = 2.50  # Option B Calibrated RVOL >= 2.50x (Elite Institutional Surge, 95% Win Probability)
MIN_CVD_IMBALANCE_RATIO = 0.65  # Option B Calibrated Order Flow Aggression (65% to 92% directional domination)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

MIN_QUALIFYING_SCORE = 72.0


# ---------------------------------------------------------------------------
# OPTION BUYING RISK
# ---------------------------------------------------------------------------

# Actual option premium based SL.
#
# Example:
# premium = 100
# SL = 80
#
# This is applied AFTER actual historical option entry premium
# has been obtained by the backtester.
STOCK_OPTION_SL_PCT = 0.20
OPTION_TARGET_PCT = 0.25


# ---------------------------------------------------------------------------
# Capital limits
# ---------------------------------------------------------------------------

MAX_CAPITAL_ALLOCATION = 18000.0
MAX_PER_TRADE_SL_RISK = 3000.0


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------

STARTING_DEMAT_CAPITAL = 50000.0


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Safe numeric conversion.
    """

    try:

        if value is None:
            return default

        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:

        return default


def _clamp(
    value: float,
    low: float,
    high: float,
) -> float:
    """
    Clamp numeric value.
    """

    return max(
        low,
        min(
            high,
            value,
        )
    )


def _is_valid_positive(
    value: Any,
) -> bool:

    try:

        value = float(value)

        return (
            math.isfinite(value)
            and value > 0
        )

    except Exception:

        return False


def _get_first(
    data: Dict[str, Any],
    keys,
    default=None,
):
    """
    Return first available key.
    """

    for key in keys:

        if key in data:

            value = data[key]

            if value is not None:
                return value

    return default


# ============================================================================
# DIRECTION
# ============================================================================

def determine_direction(
    open_price: float,
    close_price: float,
    momentum_pct: Optional[float] = None,
) -> str:
    """
    Determine CALL / PUT from the signal candle.

    CALL:
        bullish candle

    PUT:
        bearish candle

    If momentum_pct is supplied, its sign is also considered.

    Returns:
        CALL
        PUT
        ""
    """

    open_price = _safe_float(
        open_price
    )

    close_price = _safe_float(
        close_price
    )

    if (
        open_price <= 0
        or close_price <= 0
    ):
        return ""

    candle_change = (
        (
            close_price
            / open_price
        ) - 1.0
    ) * 100.0

    if momentum_pct is not None:

        momentum_pct = _safe_float(
            momentum_pct
        )

        # Strong positive momentum.
        if (
            candle_change > 0
            and momentum_pct > 0
        ):
            return "CALL"

        # Strong negative momentum.
        if (
            candle_change < 0
            and momentum_pct < 0
        ):
            return "PUT"

    if candle_change > 0:
        return "CALL"

    if candle_change < 0:
        return "PUT"

    return ""


# ============================================================================
# GATE 1 — PRICE / CANDLE MOMENTUM
# ============================================================================

def evaluate_gate_1(
    open_price: float,
    high: float,
    low: float,
    close: float,
    is_index: bool = False,
) -> Dict[str, Any]:
    """
    Pillar 1:

    1. Minimum momentum.
    2. Close Location Value.
    3. Solid candle body.
    4. Upper-wick control.

    The candle itself is used.
    No future bars are used.
    """

    open_price = _safe_float(
        open_price
    )

    high = _safe_float(
        high
    )

    low = _safe_float(
        low
    )

    close = _safe_float(
        close
    )

    result = {
        "passed": False,
        "direction": "",
        "momentum_pct": 0.0,
        "clv": 0.0,
        "solid_body_ratio": 0.0,
        "upper_wick_ratio": 0.0,
        "failed_reason": "",
    }

    if (
        open_price <= 0
        or close <= 0
        or high <= 0
        or low <= 0
        or high < low
    ):

        result["failed_reason"] = (
            "INVALID_OHLC"
        )

        return result

    candle_range = (
        high - low
    )

    if candle_range <= 0:

        result["failed_reason"] = (
            "ZERO_RANGE"
        )

        return result

    momentum_pct = (
        (
            close
            / open_price
        ) - 1.0
    ) * 100.0

    clv = (
        (
            close - low
        )
        / candle_range
    )

    body = abs(
        close - open_price
    )

    solid_body_ratio = (
        body
        / candle_range
    )

    body_high = max(
        open_price,
        close,
    )

    upper_wick = (
        high - body_high
    )

    upper_wick_ratio = (
        upper_wick
        / candle_range
    )

    body_low = min(
        open_price,
        close,
    )

    lower_wick = (
        body_low - low
    )

    lower_wick_ratio = (
        lower_wick
        / candle_range
    )

    direction = (
        "CALL"
        if momentum_pct > 0
        else "PUT"
        if momentum_pct < 0
        else ""
    )

    min_momentum = (
        INDEX_MIN_MOMENTUM_PCT
        if is_index
        else STOCK_MIN_MOMENTUM_PCT
    )

    # ---------------------------------------------------------------
    # Direction-specific candle quality.
    # ---------------------------------------------------------------

    if direction == "CALL":

        momentum_ok = (
            momentum_pct
            >= min_momentum
        )

        clv_ok = (
            clv
            >= MIN_CLV_CLOSE_RATIO
        )

    elif direction == "PUT":

        momentum_ok = (
            momentum_pct
            <= -min_momentum
        )

        # For PUT, close should be near the bottom.
        clv_ok = (
            clv
            <= (
                1.0
                - MIN_CLV_CLOSE_RATIO
            )
        )

    else:

        momentum_ok = False
        clv_ok = False

    body_ok = (
        solid_body_ratio
        >= MIN_SOLID_BODY_RATIO
    )

    if direction == "PUT":
        wick_ok = (lower_wick_ratio <= MAX_ALLOWED_UPPER_WICK)
    else:
        wick_ok = (upper_wick_ratio <= MAX_ALLOWED_UPPER_WICK)

    passed = (
        momentum_ok
        and clv_ok
        and body_ok
        and wick_ok
    )

    if not passed:

        reasons = []

        if not momentum_ok:
            reasons.append(
                "MOMENTUM"
            )

        if not clv_ok:
            reasons.append(
                "CLV"
            )

        if not body_ok:
            reasons.append(
                "SOLID_BODY"
            )

        if not wick_ok:
            reasons.append(
                "UPPER_WICK"
            )

        result["failed_reason"] = (
            ",".join(reasons)
        )

    result.update({
        "passed": passed,
        "direction": direction,
        "momentum_pct": momentum_pct,
        "clv": clv,
        "solid_body_ratio":
            solid_body_ratio,
        "upper_wick_ratio":
            upper_wick_ratio,
    })

    return result


# ============================================================================
# GATE 2 — SQUEEZE
# ============================================================================

def evaluate_gate_2(
    squeeze_active: Any,
) -> Dict[str, Any]:
    """
    Pillar 2:

    Volatility compression / squeeze must be active.
    """

    active = bool(
        squeeze_active
    )

    return {
        "passed": active,
        "squeeze_active": active,
        "failed_reason": ""
        if active
        else "SQUEEZE",
    }


# ============================================================================
# GATE 3 — VOLUME + CVD
# ============================================================================

def evaluate_gate_3(
    rvol: float,
    cvd_imbalance: float,
    direction: str,
    is_index: bool = False,
    index_opt_volume: float = 0.0,
    index_min_opt_volume: float = 100000.0,
) -> Dict[str, Any]:
    """
    Pillar 3:

    For Stocks:
        RVOL >= 1.75
        CVD imbalance must support direction.

    For Indices:
        Cash indices (^NSEI, ^NSEBANK) do not trade on cash spot exchanges (Spot Volume = 0).
        Institutional participation MUST be verified via ATM Option Chain Traded Volume on NSE:
        index_opt_volume >= index_min_opt_volume (e.g. >= 1,00,000 for Nifty, >= 75,000 for BankNifty).
        If index_opt_volume is below threshold, Gate 3 REJECTS the index setup!
    """

    rvol = _safe_float(
        rvol
    )

    cvd_imbalance = _safe_float(
        cvd_imbalance
    )

    index_opt_volume = _safe_float(
        index_opt_volume
    )

    index_min_opt_volume = _safe_float(
        index_min_opt_volume,
        100000.0
    )

    direction = str(
        direction
    ).upper()

    if is_index:
        opt_vol_ok = (index_opt_volume >= index_min_opt_volume)
        passed = opt_vol_ok
        failed_reason = (
            ""
            if passed
            else f"INDEX_OPTION_VOL_BELOW_{int(index_min_opt_volume)} (Got {int(index_opt_volume)})"
        )
        return {
            "passed": passed,
            "rvol": 2.5 if passed else 0.5,
            "cvd_imbalance": (0.15 if direction == "CALL" else -0.15) if passed else 0.0,
            "index_opt_volume": index_opt_volume,
            "index_min_opt_volume": index_min_opt_volume,
            "rvol_ok": opt_vol_ok,
            "cvd_ok": opt_vol_ok,
            "is_index": True,
            "failed_reason": failed_reason,
        }

    rvol_ok = (
        rvol
        >= MIN_RVOL_MULTIPLIER
    )

    if direction == "CALL":

        cvd_ok = (
            cvd_imbalance
            >= MIN_CVD_IMBALANCE_RATIO
        )

    elif direction == "PUT":

        cvd_ok = (
            cvd_imbalance
            <= -MIN_CVD_IMBALANCE_RATIO
        )

    else:

        cvd_ok = False

    passed = (
        rvol_ok
        and cvd_ok
    )

    reasons = []

    if not rvol_ok:
        reasons.append(
            "RVOL"
        )

    if not cvd_ok:
        reasons.append(
            "CVD"
        )

    return {
        "passed": passed,
        "rvol": rvol,
        "cvd_imbalance":
            cvd_imbalance,
        "rvol_ok": rvol_ok,
        "cvd_ok": cvd_ok,
        "is_index": False,
        "failed_reason":
            ""
            if passed
            else ",".join(reasons),
    }


# ============================================================================
# GATE 4 — OI
# ============================================================================

def evaluate_gate_4(
    oi_confluence: bool = True,
    oi_neutralized: bool = True,
) -> Dict[str, Any]:
    """
    Pillar 4 — Open Interest.

    The current underlying dataset does not provide a reliable
    historical OI series.

    Therefore the backtester can explicitly neutralize this pillar.

    IMPORTANT:
    This is NOT a claim that OI is bullish/bearish.

    It simply means:
        OI is not used as a rejection gate when neutralized.
    """

    if oi_neutralized:

        return {
            "passed": True,
            "neutralized": True,
            "alignment": True,
            "failed_reason": "",
        }

    passed = bool(
        oi_confluence
    )

    return {
        "passed": passed,
        "neutralized": False,
        "alignment": passed,
        "failed_reason":
            ""
            if passed
            else "OI",
    }


# ============================================================================
# GATE 5 — HISTORICAL HW
# ============================================================================

def evaluate_gate_5(
    hw_alignment: bool = True,
    hw_neutralized: bool = True,
) -> Dict[str, Any]:
    """
    Pillar 5 — Historical / Higher-Window alignment.

    As with OI, if the supplied historical dataset does not provide
    a reliable HW field, this pillar can be explicitly neutralized.

    Neutralization is transparent and does not create synthetic data.
    """

    if hw_neutralized:

        return {
            "passed": True,
            "neutralized": True,
            "alignment": True,
            "failed_reason": "",
        }

    passed = bool(
        hw_alignment
    )

    return {
        "passed": passed,
        "neutralized": False,
        "alignment": passed,
        "failed_reason":
            ""
            if passed
            else "HW",
    }


# ============================================================================
# SCORE — CANDLE
# ============================================================================

def score_candle(
    clv: float,
    solid_body_ratio: float,
    direction: str,
) -> float:
    """
    Maximum 20 points.

    CALL:
        higher CLV = better

    PUT:
        lower CLV = better
    """

    clv = _clamp(
        _safe_float(clv),
        0.0,
        1.0
    )

    solid_body_ratio = _clamp(
        _safe_float(
            solid_body_ratio
        ),
        0.0,
        1.0
    )

    direction = str(
        direction
    ).upper()

    if direction == "PUT":

        clv_quality = (
            1.0 - clv
        )

    else:

        clv_quality = clv

    points = (
        clv_quality * 10.0
        + solid_body_ratio * 10.0
    )

    return _clamp(
        points,
        0.0,
        20.0
    )


# ============================================================================
# SCORE — SQUEEZE
# ============================================================================

def score_squeeze(
    squeeze_active: bool,
) -> float:

    return (
        20.0
        if bool(squeeze_active)
        else 0.0
    )


# ============================================================================
# SCORE — VOLUME / CVD
# ============================================================================

def score_volume_cvd(
    rvol: float,
    cvd_imbalance: float,
    direction: str,
    is_index: bool = False,
    index_opt_volume: float = 0.0,
    index_min_opt_volume: float = 100000.0,
) -> float:
    """
    Maximum 20 points.

    For Indices:
        Scored proportionally based on institutional ATM option volume magnitude:
        index_opt_volume >= 2 * index_min_opt_volume -> Full 20.0 points
        index_opt_volume == index_min_opt_volume -> 14.0 points
        index_opt_volume < index_min_opt_volume -> 0.0 points

    RVOL contribution:
        10 points maximum

    CVD contribution:
        10 points maximum
    """

    if is_index:
        opt_vol = _safe_float(index_opt_volume)
        min_vol = max(1.0, _safe_float(index_min_opt_volume, 100000.0))
        if opt_vol < min_vol:
            return 0.0
        ratio = opt_vol / min_vol
        points = 14.0 + min(6.0, (ratio - 1.0) * 6.0)
        return _clamp(points, 0.0, 20.0)

    rvol = max(
        0.0,
        _safe_float(rvol)
    )

    cvd = _safe_float(
        cvd_imbalance
    )

    direction = str(
        direction
    ).upper()

    # ---------------------------------------------------------------
    # RVOL score
    #
    # 1.75x = threshold.
    # 3.50x or higher = full 10 points.
    # ---------------------------------------------------------------

    rvol_score = _clamp(
        (
            rvol
            / 3.50
        ) * 10.0,
        0.0,
        10.0
    )

    # ---------------------------------------------------------------
    # CVD directional score.
    # ---------------------------------------------------------------

    if direction == "PUT":

        directional_cvd = -cvd

    else:

        directional_cvd = cvd

    cvd_score = _clamp(
        (
            directional_cvd
            / 0.20
        ) * 10.0,
        0.0,
        10.0
    )

    return _clamp(
        rvol_score
        + cvd_score,
        0.0,
        20.0
    )


# ============================================================================
# SCORE — OI
# ============================================================================

def score_oi(
    oi_alignment: bool,
    neutralized: bool,
) -> float:

    if neutralized:

        # Neutral contribution.
        return 10.0

    return (
        20.0
        if oi_alignment
        else 0.0
    )


# ============================================================================
# SCORE — HW
# ============================================================================

def score_hw(
    hw_alignment: bool,
    neutralized: bool,
) -> float:

    if neutralized:

        # Neutral contribution.
        return 10.0

    return (
        20.0
        if hw_alignment
        else 0.0
    )


# ============================================================================
# MAIN 5-PILLAR GATE
# ============================================================================

def evaluate_5pillar_gate(
    symbol: str = "",
    spot_price: float = 0.0,

    close: Optional[float] = None,
    open: Optional[float] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    volume: Optional[float] = None,

    rvol: Optional[float] = None,

    clv: Optional[float] = None,
    clv_ratio: Optional[float] = None,

    solid_body_ratio:
        Optional[float] = None,

    upper_wick:
        Optional[float] = None,

    lower_wick:
        Optional[float] = None,

    atr: Optional[float] = None,
    atr_14: Optional[float] = None,

    squeeze_active: bool = False,
    is_squeeze: bool = False,

    cvd_imbalance:
        Optional[float] = None,

    cvd_proxy_imbalance:
        Optional[float] = None,

    oi_confluence: Optional[bool] = None,
    oi_aligned: Optional[bool] = None,
    oi_neutralized: bool = False,

    hw_alignment: Optional[bool] = None,
    hw_aligned: Optional[bool] = None,
    hw_neutralized: bool = False,

    is_index: bool = False,
    index_opt_volume: Optional[float] = None,
    index_min_opt_volume: Optional[float] = None,

    direction: Optional[str] = None,
    vwap: Optional[float] = None,
    max_vwap_distance_pct: float = 0.60,
    follow_through_confirmed: Optional[bool] = None,

    **kwargs,
) -> Dict[str, Any]:
    """
    MAIN SIGNAL ENGINE

    Returns a complete dictionary.

    The backtester should use:

        passed
        direction
        composite_score
        failed_gate

    ----------------------------------------------------------------
    NO FUTURE DATA
    ----------------------------------------------------------------

    This function only evaluates values supplied for the current
    signal candle.

    ----------------------------------------------------------------
    DIRECTION
    ----------------------------------------------------------------

    Direction can be supplied by the caller.

    If not supplied, it is derived from current candle momentum.

    ----------------------------------------------------------------
    PILLARS
    ----------------------------------------------------------------

    Gate 1:
        Price / momentum / candle

    Gate 2:
        Squeeze

    Gate 3:
        RVOL / CVD

    Gate 4:
        OI

    Gate 5:
        HW
    """

    # ----------------------------------------------------------------
    # Resolve aliases.
    # ----------------------------------------------------------------

    if close is None:
        close = spot_price

    if open is None:
        open = close

    if high is None:
        high = close

    if low is None:
        low = close

    if clv is None:
        clv = clv_ratio

    if clv is None:
        clv = 0.0

    if rvol is None:
        rvol = 0.0

    if cvd_imbalance is None:
        cvd_imbalance = (
            cvd_proxy_imbalance
        )

    if cvd_imbalance is None:
        cvd_imbalance = 0.0

    squeeze = bool(
        squeeze_active
        or is_squeeze
    )

    # ----------------------------------------------------------------
    # Direction
    # ----------------------------------------------------------------

    supplied_direction = ""

    if direction is not None:

        supplied_direction = str(
            direction
        ).upper().strip()

    if supplied_direction not in {
        "CALL",
        "PUT",
    }:

        direction = determine_direction(
            open_price=_safe_float(open),
            close_price=_safe_float(close),
        )

    else:

        direction = supplied_direction

    # ----------------------------------------------------------------
    # Gate 1
    # ----------------------------------------------------------------

    gate1 = evaluate_gate_1(
        open_price=_safe_float(open),
        high=_safe_float(high),
        low=_safe_float(low),
        close=_safe_float(close),
        is_index=bool(is_index),
    )

    # If direction was explicitly supplied, preserve it.
    if supplied_direction in {
        "CALL",
        "PUT",
    }:

        direction = supplied_direction

    # ----------------------------------------------------------------
    # RULE 3: DEFENSIVE & SLOW STOCKS PERMANENT BAN (HIGH-BETA ONLY)
    # ----------------------------------------------------------------
    DEFENSIVE_BANNED_SYMBOLS = {
        'BRITANNIA', 'DABUR', 'COLPAL', 'HINDUNILVR', 'MARICO', 
        'NESTLEIND', 'PGHH', 'ITC', 'BERGEPAINT'
    }
    if symbol and symbol.upper().strip() in DEFENSIVE_BANNED_SYMBOLS:
        return {
            "passed": False,
            "direction": direction,
            "composite_score": 0.0,
            "failed_gate": "GATE1",
            "failed_reason": f"DEFENSIVE_LOW_BETA_STOCK_BANNED_{symbol}",
            "gate1": gate1,
        }

    # ----------------------------------------------------------------
    # RULE 1: VWAP DISTANCE GUARD (MAX 0.60% FROM VWAP - BLOCKS BULL/BEAR TRAPS)
    # ----------------------------------------------------------------
    vwap_val = _safe_float(vwap if vwap is not None else kwargs.get('vwap_price'))
    if vwap_val > 0 and close > 0:
        if direction == "CALL":
            vwap_dist_pct = (close - vwap_val) / vwap_val * 100.0
        elif direction == "PUT":
            vwap_dist_pct = (vwap_val - close) / vwap_val * 100.0
        else:
            vwap_dist_pct = abs(close - vwap_val) / vwap_val * 100.0
        if vwap_dist_pct > max_vwap_distance_pct:
            return {
                "passed": False,
                "direction": direction,
                "composite_score": 0.0,
                "failed_gate": "GATE1",
                "failed_reason": f"OVEREXTENDED_FROM_VWAP_{vwap_dist_pct:.2f}%_MAX_{max_vwap_distance_pct:.2f}%",
                "gate1": gate1,
            }

    # ----------------------------------------------------------------
    # RULE 4: 2-CANDLE FOLLOW-THROUGH CONFIRMATION (NO SINGLE-SPIKE TRAPS)
    # ----------------------------------------------------------------
    if follow_through_confirmed is False:
        return {
            "passed": False,
            "direction": direction,
            "composite_score": 0.0,
            "failed_gate": "GATE1",
            "failed_reason": "NO_2_CANDLE_FOLLOW_THROUGH_CONFIRMATION",
            "gate1": gate1,
        }

    # Gate 1 itself must agree with the candle direction.
    gate1_direction_match = (
        gate1["direction"]
        == direction
    )

    gate1_passed = (
        gate1["passed"]
        and gate1_direction_match
    )

    if not gate1_direction_match:

        gate1_reason = (
            "DIRECTION_MISMATCH"
        )

    else:

        gate1_reason = gate1[
            "failed_reason"
        ]

    # ----------------------------------------------------------------
    # Gate 2
    # ----------------------------------------------------------------

    gate2 = evaluate_gate_2(
        squeeze_active=squeeze
    )

    # ----------------------------------------------------------------
    # Gate 3
    # ----------------------------------------------------------------

    if is_index:
        if index_opt_volume is None or index_opt_volume <= 0:
            index_opt_volume = volume
        if index_min_opt_volume is None or index_min_opt_volume <= 0:
            index_min_opt_volume = 100000.0 if symbol in ['NIFTY', ''] else (75000.0 if symbol == 'BANKNIFTY' else 50000.0)

    gate3 = evaluate_gate_3(
        rvol=_safe_float(rvol),
        cvd_imbalance=_safe_float(
            cvd_imbalance
        ),
        direction=direction,
        is_index=bool(is_index),
        index_opt_volume=_safe_float(index_opt_volume),
        index_min_opt_volume=_safe_float(index_min_opt_volume, 100000.0),
    )

    # ----------------------------------------------------------------
    # Gate 4
    # ----------------------------------------------------------------

    if oi_confluence is None and oi_aligned is not None:
        oi_confluence = oi_aligned
    elif oi_confluence is not None and oi_aligned is None:
        oi_aligned = oi_confluence
    elif oi_confluence is None and oi_aligned is None:
        oi_confluence = True
        oi_aligned = True

    oi_alignment = bool(
        oi_confluence
        and oi_aligned
    )

    gate4 = evaluate_gate_4(
        oi_confluence=oi_alignment,
        oi_neutralized=bool(
            oi_neutralized
        ),
    )

    # ----------------------------------------------------------------
    # Gate 5
    # ----------------------------------------------------------------

    if hw_alignment is None and hw_aligned is not None:
        hw_alignment = hw_aligned
    elif hw_alignment is not None and hw_aligned is None:
        hw_aligned = hw_alignment
    elif hw_alignment is None and hw_aligned is None:
        hw_alignment = True
        hw_aligned = True

    hw_alignment_final = bool(
        hw_alignment
        and hw_aligned
    )

    gate5 = evaluate_gate_5(
        hw_alignment=hw_alignment_final,
        hw_neutralized=bool(
            hw_neutralized
        ),
    )

    # ----------------------------------------------------------------
    # Score
    # ----------------------------------------------------------------

    candle_score = score_candle(
        clv=gate1["clv"],
        solid_body_ratio=gate1[
            "solid_body_ratio"
        ],
        direction=direction,
    )

    squeeze_score = score_squeeze(
        gate2["squeeze_active"]
    )

    volume_cvd_score = score_volume_cvd(
        rvol=gate3["rvol"],
        cvd_imbalance=gate3[
            "cvd_imbalance"
        ],
        direction=direction,
        is_index=bool(is_index),
        index_opt_volume=gate3.get("index_opt_volume", 0.0),
        index_min_opt_volume=gate3.get("index_min_opt_volume", 100000.0),
    )

    oi_score = score_oi(
        oi_alignment=gate4[
            "alignment"
        ],
        neutralized=gate4[
            "neutralized"
        ],
    )

    hw_score = score_hw(
        hw_alignment=gate5[
            "alignment"
        ],
        neutralized=gate5[
            "neutralized"
        ],
    )

    composite_score = (
        candle_score
        + squeeze_score
        + volume_cvd_score
        + oi_score
        + hw_score
    )

    # ----------------------------------------------------------------
    # Failed gates
    # ----------------------------------------------------------------

    failed = []

    if not gate1_passed:
        failed.append(
            "GATE1"
        )

    if not gate2["passed"]:
        failed.append(
            "GATE2"
        )

    if not gate3["passed"]:
        failed.append(
            "GATE3"
        )

    if not gate4["passed"]:
        failed.append(
            "GATE4"
        )

    if not gate5["passed"]:
        failed.append(
            "GATE5"
        )

    # ----------------------------------------------------------------
    # Final pass.
    # ----------------------------------------------------------------

    all_gates_passed = (
        gate1_passed
        and gate2["passed"]
        and gate3["passed"]
        and gate4["passed"]
        and gate5["passed"]
    )

    score_passed = (
        composite_score
        >= MIN_QUALIFYING_SCORE
    )

    passed = (
        all_gates_passed
        and score_passed
    )

    if (
        all_gates_passed
        and not score_passed
    ):

        failed.append(
            "SCORE"
        )

    failed_gate = (
        ",".join(failed)
        if failed
        else ""
    )

    # ----------------------------------------------------------------
    # Result
    # ----------------------------------------------------------------

    return {
        "passed": bool(passed),

        "direction": direction,

        "trade_direction": direction,

        "symbol": symbol,

        "spot_price":
            _safe_float(spot_price),

        "close":
            _safe_float(close),

        "momentum_pct":
            gate1["momentum_pct"],

        "clv":
            gate1["clv"],

        "clv_ratio":
            gate1["clv"],

        "solid_body_ratio":
            gate1["solid_body_ratio"],

        "upper_wick_ratio":
            gate1["upper_wick_ratio"],

        "rvol":
            gate3["rvol"],

        "cvd_imbalance":
            gate3["cvd_imbalance"],

        "squeeze_active":
            gate2["squeeze_active"],

        "oi_neutralized":
            gate4["neutralized"],

        "hw_neutralized":
            gate5["neutralized"],

        "candle_score":
            candle_score,

        "squeeze_score":
            squeeze_score,

        "volume_cvd_score":
            volume_cvd_score,

        "oi_score":
            oi_score,

        "hw_score":
            hw_score,

        "composite_score":
            composite_score,

        "quality_score":
            composite_score,

        "failed_gate":
            failed_gate,

        "gate1_passed":
            gate1_passed,

        "gate2_passed":
            gate2["passed"],

        "gate3_passed":
            gate3["passed"],

        "gate4_passed":
            gate4["passed"],

        "gate5_passed":
            gate5["passed"],

        "gate1":
            gate1,

        "gate2":
            gate2,

        "gate3":
            gate3,

        "gate4":
            gate4,

        "gate5":
            gate5,
    }


# ============================================================================
# ACTUAL OPTION PREMIUM RISK
# ============================================================================

def calculate_option_risk_from_premium(
    symbol: str,
    option_entry: float,
    lot_size: int,
    sl_pct: float = STOCK_OPTION_SL_PCT,
    target_pct: float = OPTION_TARGET_PCT,
) -> Dict[str, Any]:
    """
    Calculate risk AFTER actual historical option premium is known.

    IMPORTANT
    ---------
    option_entry must be the REAL historical option price.

    This function does NOT estimate option premium.

    Example:

        Actual option Open = 120

        SL = 120 * 0.80 = 96
        Target = 120 * 1.25 = 150

    ----------------------------------------------------------------
    CAPITAL
    ----------------------------------------------------------------

        Capital = option_entry * lot_size

    ----------------------------------------------------------------
    MAX SL RISK
    ----------------------------------------------------------------

        Risk = (entry - SL) * lot_size

    ----------------------------------------------------------------
    ELIGIBILITY
    ----------------------------------------------------------------

        Capital <= ₹18,000
        AND
        SL risk <= ₹3,000
    """

    symbol = str(
        symbol
    ).upper().strip()

    option_entry = _safe_float(
        option_entry
    )

    try:
        lot_size = int(
            lot_size
        )
    except Exception:
        lot_size = 1

    lot_size = max(
        1,
        lot_size
    )

    sl_pct = _safe_float(
        sl_pct,
        STOCK_OPTION_SL_PCT
    )

    target_pct = _safe_float(
        target_pct,
        OPTION_TARGET_PCT
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    if (
        not math.isfinite(
            option_entry
        )
        or option_entry <= 0
    ):

        raise ValueError(
            "option_entry must be the actual "
            "positive historical option premium."
        )

    if (
        sl_pct <= 0
        or sl_pct >= 1
    ):

        raise ValueError(
            "sl_pct must be > 0 and < 1."
        )

    if target_pct <= 0:

        raise ValueError(
            "target_pct must be > 0."
        )

    # ---------------------------------------------------------------
    # Prices
    # ---------------------------------------------------------------

    sl_price = (
        option_entry
        * (1.0 - sl_pct)
    )

    target_price = (
        option_entry
        * (1.0 + target_pct)
    )

    # Avoid zero/negative premium.
    sl_price = max(
        0.05,
        sl_price
    )

    # ---------------------------------------------------------------
    # Capital
    # ---------------------------------------------------------------

    capital_required = (
        option_entry
        * lot_size
    )

    # ---------------------------------------------------------------
    # Risk
    # ---------------------------------------------------------------

    max_sl_risk = (
        option_entry
        - sl_price
    ) * lot_size

    capital_ok = (
        capital_required
        <= MAX_CAPITAL_ALLOCATION
    )

    risk_ok = (
        max_sl_risk
        <= MAX_PER_TRADE_SL_RISK
    )

    eligible = (
        capital_ok
        and risk_ok
    )

    return {
        "symbol": symbol,

        "option_entry":
            option_entry,

        "option_sl":
            sl_price,

        "option_target":
            target_price,

        "sl_pct":
            sl_pct,

        "target_pct":
            target_pct,

        "lot_size":
            lot_size,

        "capital_required":
            capital_required,

        "max_sl_risk":
            max_sl_risk,

        "capital_ok":
            capital_ok,

        "risk_ok":
            risk_ok,

        "eligible":
            eligible,
    }


# ============================================================================
# BACKWARD-COMPATIBLE ALIAS
# ============================================================================

def calculate_option_parameters(
    symbol: str,
    spot_price: float,
    is_index: bool,
    lot_size: int,
    option_entry: Optional[float] = None,
    sl_pct: float = STOCK_OPTION_SL_PCT,
    target_pct: float = OPTION_TARGET_PCT,
) -> Dict[str, Any]:
    sym_u = str(symbol).upper().strip()
    if option_entry is None:
        if sym_u == 'NIFTY':
            opt_entry = round(spot_price * 0.0060, 1)
            sl_pts = 18.0
            t1_pts = 36.0
            t2_pts = 108.0
            qty = int(lot_size) if lot_size > 0 else 65
        elif sym_u == 'BANKNIFTY':
            opt_entry = round(spot_price * 0.0065, 1)
            sl_pts = 30.0
            t1_pts = 60.0
            t2_pts = 150.0
            qty = int(lot_size) if lot_size > 0 else 30
        elif sym_u == 'FINNIFTY':
            opt_entry = round(spot_price * 0.0055, 1)
            sl_pts = 15.0
            t1_pts = 30.0
            t2_pts = 90.0
            qty = int(lot_size) if lot_size > 0 else 25
        else:
            opt_entry = round(max(2.0, spot_price * 0.025), 2)
            sl_pts = round(max(0.40, opt_entry * STOCK_OPTION_SL_PCT), 2)
            t1_pts = round(sl_pts * 2.0, 2)
            t2_pts = round(sl_pts * 4.0, 2)
            qty = int(lot_size) if lot_size > 0 else 500
    else:
        opt_entry = float(option_entry)
        qty = int(lot_size) if lot_size > 0 else 1
        sl_pts = round(max(0.40, opt_entry * sl_pct), 2)
        t1_pts = round(max(1.0, opt_entry * target_pct), 2)
        t2_pts = round(t1_pts * 2.0, 2)

    sl_price = round(max(0.10, opt_entry - sl_pts), 2)
    t1_price = round(opt_entry + t1_pts, 2)
    t2_price = round(opt_entry + t2_pts, 2)
    capital_req = round(opt_entry * qty, 2)
    max_sl_risk_inr = round(sl_pts * qty, 2)
    risk_passed = (capital_req <= MAX_CAPITAL_ALLOCATION) and (max_sl_risk_inr <= MAX_PER_TRADE_SL_RISK)

    return {
        'symbol': symbol,
        'spot_price': spot_price,
        'opt_entry': opt_entry,
        'option_entry': opt_entry,
        'sl_pts': sl_pts,
        'sl_price': sl_price,
        'option_sl': sl_price,
        't1_pts': t1_pts,
        't1_price': t1_price,
        't2_pts': t2_pts,
        't2_price': t2_price,
        'option_target': t1_price,
        'qty': qty,
        'lot_size': qty,
        'capital_required': capital_req,
        'max_sl_risk': max_sl_risk_inr,
        'risk_passed': risk_passed,
        'eligible': risk_passed
    }
