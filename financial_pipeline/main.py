"""
Market Pulse Data Pipeline — Orchestrator
==========================================

Executes four deterministic data pillars sequentially with full error isolation,
then serialises the consolidated result to:

    /app/output/market_pulse_payload.json   ← structured machine-readable payload
    /app/output/raw_metrics.md              ← human-readable Markdown summary

Usage:
    python main.py                           # normal run
    OUTPUT_DIR=/tmp/out python main.py       # override output directory

Exit codes:
    0 — all pillars succeeded
    1 — one or more pillars raised an exception (partial output still written)
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Local imports ─────────────────────────────────────────────────────────────
from utils.logger import configure_logging
from pillars import cot, cta, gex, sector

OUTPUT_DIR    = Path(os.getenv("OUTPUT_DIR", "/app/output"))
PAYLOAD_FILE  = OUTPUT_DIR / "market_pulse_payload.json"
MARKDOWN_FILE = OUTPUT_DIR / "raw_metrics.md"

PILLAR_REGISTRY: list[tuple[str, Any]] = [
    ("gex",    gex.run),
    ("cta",    cta.run),
    ("cot",    cot.run),
    ("sector", sector.run),
]

logger = logging.getLogger(__name__)


# ── Utility helpers ───────────────────────────────────────────────────────────

def _nan_to_none(obj: Any) -> Any:
    """Recursively replace float NaN / ±Inf with JSON-legal None."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _fmt(val: Any, fmt: str = ".2f") -> str:
    """Format a number, returning 'N/A' for None / NaN."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    try:
        return format(val, fmt)
    except (TypeError, ValueError):
        return str(val)


# ── Pillar runner (error-isolated) ───────────────────────────────────────────

def _run_pillar(name: str, fn: Any) -> dict[str, Any]:
    """
    Execute a single pillar function.  Any exception is caught, logged, and
    returned as a structured error dict so the orchestrator continues.
    """
    t0 = time.monotonic()
    try:
        result  = fn()
        elapsed = time.monotonic() - t0
        logger.info("Pillar [%-6s] OK   (%.2f s)", name, elapsed)
        return {"status": "ok", "elapsed_s": round(elapsed, 3), "data": result}
    except Exception:
        elapsed = time.monotonic() - t0
        tb      = traceback.format_exc()
        logger.error("Pillar [%-6s] FAIL (%.2f s)\n%s", name, elapsed, tb)
        return {"status": "error", "elapsed_s": round(elapsed, 3), "error": tb, "data": None}


# ── Markdown renderer ─────────────────────────────────────────────────────────

def _render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    ts = payload.get("run_timestamp", "unknown")

    lines += [
        "# Market Pulse — Raw Metrics",
        "",
        f"**Run timestamp (UTC):** {ts}  ",
        f"**Schema version:** {payload.get('schema_version', '?')}",
        "",
        "---",
    ]

    pillars = payload.get("pillars", {})

    # ── GEX ──────────────────────────────────────────────────────────────────
    lines += ["", "## Pillar 1 — Dealer Gamma Exposure (GEX)"]
    gp = pillars.get("gex", {})
    if gp.get("status") == "ok" and gp.get("data"):
        d = gp["data"]
        lines += [
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Ticker | `{d.get('ticker')}` |",
            f"| Spot Price | {_fmt(d.get('spot_price'), ',.2f')} |",
            f"| Call Wall | {_fmt(d.get('call_wall'), ',.2f')} |",
            f"| Put Wall | {_fmt(d.get('put_wall'), ',.2f')} |",
            f"| GEX Flip Zone | {_fmt(d.get('gex_flip_zone'), ',.2f')} |",
            f"| Total Net GEX ($M) | {_fmt((d.get('total_net_gex') or 0) / 1e6, ',.3f')} |",
            "",
            "### Top 5 Strikes by |Net GEX|",
            "",
            "| Strike | Net GEX ($) | Call GEX ($) | Put GEX ($) | Call OI | Put OI |",
            "|--------|-------------|--------------|-------------|---------|--------|",
        ]
        for s in d.get("top_5_strikes", []):
            lines.append(
                f"| {_fmt(s['strike'], ',.0f')} "
                f"| {_fmt(s['net_dollar_gex'], ',.0f')} "
                f"| {_fmt(s['call_dollar_gex'], ',.0f')} "
                f"| {_fmt(s['put_dollar_gex'], ',.0f')} "
                f"| {s['call_oi']:,} "
                f"| {s['put_oi']:,} |"
            )
    else:
        err = (gp.get("error") or "unknown error")[:300]
        lines.append(f"\n> **ERROR:** `{err}`")

    # ── CTA ───────────────────────────────────────────────────────────────────
    lines += ["", "## Pillar 2 — CTA / Systematic Flow"]
    cp = pillars.get("cta", {})
    if cp.get("status") == "ok" and cp.get("data"):
        lines += [
            "",
            "| Ticker | Name | Signal | Ret 12M | Vol Ann | Sharpe | >50SMA | >200SMA | BrkHi100 | BrkLo20 |",
            "|--------|------|--------|---------|---------|--------|--------|---------|----------|---------|",
        ]
        for a in cp["data"].get("assets", []):
            if "trend_signal" not in a:
                lines.append(f"| {a.get('ticker')} | {a.get('name')} | ⚠ ERROR | — | — | — | — | — | — | — |")
                continue
            t   = "✓" if a.get("is_above_50sma")     else "✗"
            tw  = "✓" if a.get("is_above_200sma")    else "✗"
            bh  = "✓" if a.get("is_breakout_high_100") else "✗"
            bl  = "✓" if a.get("is_breakout_low_20")  else "✗"
            lines.append(
                f"| `{a['ticker']}` | {a['name']} | **{a['trend_signal']}** "
                f"| {_fmt(a['return_12m'] * 100, '+.1f')}% "
                f"| {_fmt(a['volatility_annualized'] * 100, '.1f')}% "
                f"| {_fmt(a['sharpe_12m'], '+.2f')} "
                f"| {t} | {tw} | {bh} | {bl} |"
            )
    else:
        err = (cp.get("error") or "unknown error")[:300]
        lines.append(f"\n> **ERROR:** `{err}`")

    # ── COT ───────────────────────────────────────────────────────────────────
    lines += ["", "## Pillar 3 — COT Managed Money Positioning"]
    ctp = pillars.get("cot", {})
    if ctp.get("status") == "ok" and ctp.get("data"):
        lines += [
            "",
            "| Asset | As-Of | MM Longs | MM Shorts | Net Contracts | Z-Score 52W | Extreme |",
            "|-------|-------|----------|-----------|---------------|-------------|---------|",
        ]
        for a in ctp["data"].get("assets", []):
            if "mm_net" not in a:
                lines.append(f"| {a.get('name')} | — | — | — | — | — | ⚠ ERROR |")
                continue
            z_str   = _fmt(a.get("zscore_52w"), "+.2f")
            extreme = "⚠ **EXTREME**" if a.get("is_extreme_long") or a.get("is_extreme_short") else ""
            direction = "LONG" if a.get("is_extreme_long") else ("SHORT" if a.get("is_extreme_short") else "")
            lines.append(
                f"| {a['name']} | {a['as_of_date']} "
                f"| {a['mm_longs']:,} | {a['mm_shorts']:,} | {a['mm_net']:+,} "
                f"| {z_str} | {extreme} {direction} |"
            )
    else:
        err = (ctp.get("error") or "unknown error")[:300]
        lines.append(f"\n> **ERROR:** `{err}`")

    # ── Sector Rotation ───────────────────────────────────────────────────────
    lines += ["", "## Pillar 4 — Sector Rotation Matrix"]
    sp = pillars.get("sector", {})
    if sp.get("status") == "ok" and sp.get("data"):
        d = sp["data"]
        lines += [
            "",
            f"**Regime:** `{d.get('rotation_signal')}`  |  "
            f"**Leading:** {d.get('top_sector')}  |  "
            f"**Lagging:** {d.get('bottom_sector')}",
            "",
            f"_SPY reference: 5d={_fmt(d.get('spy_ret_5d_pct'), '+.2f')}%  "
            f"20d={_fmt(d.get('spy_ret_20d_pct'), '+.2f')}%  "
            f"60d={_fmt(d.get('spy_ret_60d_pct'), '+.2f')}%_",
            "",
            "| # | Ticker | Sector | Signal | Score | Rel5d | Rel20d | Rel60d | Accel |",
            "|---|--------|--------|--------|-------|-------|--------|--------|-------|",
        ]
        for e in d.get("leaderboard", []):
            lines.append(
                f"| {e['rank']} | `{e['ticker']}` | {e['sector']} | **{e['signal']}** "
                f"| {_fmt(e['composite_score_pct'], '+.2f')}% "
                f"| {_fmt(e['rel_5d_pct'], '+.2f')}% "
                f"| {_fmt(e['rel_20d_pct'], '+.2f')}% "
                f"| {_fmt(e['rel_60d_pct'], '+.2f')}% "
                f"| {_fmt(e['momentum_acceleration_pct'], '+.2f')}% |"
            )
    else:
        err = (sp.get("error") or "unknown error")[:300]
        lines.append(f"\n> **ERROR:** `{err}`")

    lines += [
        "",
        "---",
        "*All values computed deterministically — no AI synthesis. Pure math on market data.*",
        "",
    ]
    return "\n".join(lines)


# ── Main orchestrator ─────────────────────────────────────────────────────────

def main() -> int:
    configure_logging()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(tz=timezone.utc).isoformat()

    logger.info("=" * 70)
    logger.info("Market Pulse Pipeline  —  %s", run_ts)
    logger.info("Output dir: %s", OUTPUT_DIR)
    logger.info("=" * 70)

    pillar_results: dict[str, Any] = {}
    for name, fn in PILLAR_REGISTRY:
        logger.info("── [%s] ──", name.upper())
        pillar_results[name] = _run_pillar(name, fn)

    ok_pillars    = [k for k, v in pillar_results.items() if v["status"] == "ok"]
    error_pillars = [k for k, v in pillar_results.items() if v["status"] == "error"]

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "run_timestamp":  run_ts,
        "summary": {
            "pillars_ok":    ok_pillars,
            "pillars_error": error_pillars,
            "total_elapsed_s": round(sum(v["elapsed_s"] for v in pillar_results.values()), 3),
        },
        "pillars": pillar_results,
    }

    safe_payload = _nan_to_none(payload)

    with open(PAYLOAD_FILE, "w", encoding="utf-8") as fh:
        json.dump(safe_payload, fh, indent=2, default=str, ensure_ascii=False)
    logger.info("JSON payload  → %s  (%d bytes)", PAYLOAD_FILE, PAYLOAD_FILE.stat().st_size)

    md_text = _render_markdown(safe_payload)
    with open(MARKDOWN_FILE, "w", encoding="utf-8") as fh:
        fh.write(md_text)
    logger.info("Markdown      → %s  (%d bytes)", MARKDOWN_FILE, MARKDOWN_FILE.stat().st_size)

    if error_pillars:
        logger.warning("Pipeline finished with errors in pillars: %s", error_pillars)
        return 1

    logger.info("Pipeline complete — all %d pillars OK.", len(ok_pillars))
    return 0


if __name__ == "__main__":
    sys.exit(main())
