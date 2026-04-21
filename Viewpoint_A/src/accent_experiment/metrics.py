from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


def _levenshtein(ref: list[str], hyp: list[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)

    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        curr = [i]
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _levenshtein(ref_chars, hyp_chars) / len(ref_chars)


@dataclass
class OverallMetrics:
    overall_wer: float
    overall_cer: float
    macro_wer: float
    macro_cer: float
    worst_group_wer: float
    worst_group_cer: float
    majority_minority_gap_wer: float


def summarize_metrics(pred_df: pd.DataFrame, majority_accents: list[str]) -> tuple[pd.DataFrame, OverallMetrics]:
    df = pred_df.copy()
    df["wer"] = df.apply(lambda r: wer(str(r["reference"]), str(r["prediction"])), axis=1)
    df["cer"] = df.apply(lambda r: cer(str(r["reference"]), str(r["prediction"])), axis=1)

    by_accent = (
        df.groupby("accent", dropna=False)
        .agg(
            samples=("sample_id", "count"),
            wer=("wer", "mean"),
            cer=("cer", "mean"),
        )
        .reset_index()
        .sort_values("wer", ascending=False)
    )

    overall_wer = float(df["wer"].mean())
    overall_cer = float(df["cer"].mean())
    macro_wer = float(by_accent["wer"].mean()) if not by_accent.empty else math.nan
    macro_cer = float(by_accent["cer"].mean()) if not by_accent.empty else math.nan
    worst_group_wer = float(by_accent["wer"].max()) if not by_accent.empty else math.nan
    worst_group_cer = float(by_accent["cer"].max()) if not by_accent.empty else math.nan

    majority_mask = by_accent["accent"].isin(majority_accents)
    majority_wer = float(by_accent[majority_mask]["wer"].mean()) if majority_mask.any() else math.nan
    minority_wer = float(by_accent[~majority_mask]["wer"].mean()) if (~majority_mask).any() else math.nan
    gap = majority_wer - minority_wer if not math.isnan(majority_wer) and not math.isnan(minority_wer) else math.nan

    overall = OverallMetrics(
        overall_wer=overall_wer,
        overall_cer=overall_cer,
        macro_wer=macro_wer,
        macro_cer=macro_cer,
        worst_group_wer=worst_group_wer,
        worst_group_cer=worst_group_cer,
        majority_minority_gap_wer=gap,
    )
    return by_accent, overall
