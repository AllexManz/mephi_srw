"""Module 7 — lightweight dataset screening before training."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


def label_proxy(example: Dict[str, str]) -> str:
    """Cheap pseudo-label extracted from instruction text."""
    if "instruction" in example:
        for token in example["instruction"].split():
            if token.upper().startswith("CVE-"):
                return "cve"
        low = example.get("instruction", "").lower()
        if "уязвим" in low:
            return "cve"
        if "тактик" in low:
            return "mitre"
        if "техник" in low:
            return "technique"
    return "misc"


def label_distribution_uniformity(labels: Sequence[str]) -> Dict[str, Any]:
    ctr = Counter(labels)
    cats = sorted(ctr.keys())
    counts = np.array([ctr[c] for c in cats])
    if counts.sum() == 0 or len(cats) < 2:
        return {"statistic": 0.0, "pvalue": None, "categories": dict(ctr)}
    expected = np.full_like(counts, fill_value=counts.sum() / len(cats), dtype=float)
    stat, p = stats.chisquare(counts, f_exp=expected)
    return {"statistic": float(stat), "pvalue": float(p), "categories": dict(ctr)}


class TrainingDataValidator:
    """Statistical probes + textual outlier proxies for ingestion QA."""

    def __init__(
        self,
        reference_manifest: Dict[str, Any] | None = None,
        max_iso_fraction_threshold: float = 0.15,
    ):
        self.reference_manifest = reference_manifest or {}
        self.max_iso_fraction_threshold = max_iso_fraction_threshold

    def isolation_style_scores(self, corpus: Sequence[str]) -> Tuple[np.ndarray, Dict[str, float]]:
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return np.zeros(len(corpus), dtype=np.float32), {"note": "sklearn_missing"}

        if len(corpus) < 5:
            return np.zeros(len(corpus)), {"note": "too_few_documents"}

        vectorizer = TfidfVectorizer(max_features=4096)
        tfidf = vectorizer.fit_transform(corpus)
        forest = IsolationForest(random_state=0, contamination=max(5 / tfidf.shape[0], 0.002))
        preds = forest.fit_predict(tfidf)
        outliers = preds == -1
        frac = float(outliers.mean())
        meta = {"isolation_fraction": frac, "method": "isolation_forest_tfidf"}
        return outliers.astype(np.float32), meta

    def compare_to_reference(self, current_manifest: Dict[str, Any], prev_manifest_path: Path) -> Dict[str, Any]:
        prev = {}
        if prev_manifest_path.is_file():
            with open(prev_manifest_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
        curr_cats = Counter(current_manifest.get("label_buckets", {}))
        prev_cats = Counter(prev.get("label_buckets", {}))
        frac_diff = {}
        for key in curr_cats:
            frac_diff[key] = abs(
                curr_cats[key] / max(sum(curr_cats.values()), 1)
                - prev_cats[key] / max(sum(prev_cats.values()), 1)
            )
        max_shift = float(max(frac_diff.values()) if frac_diff else 0.0)
        return {"max_distribution_shift_bucket": max_shift, "pairs": frac_diff, "flag": max_shift > 0.05}

    def summarize_dataset(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [label_proxy(ex) for ex in examples]
        chi = label_distribution_uniformity(labels)
        corp = [(ex.get("instruction", "") + " " + ex.get("input", ""))[:2000] for ex in examples]
        iso_flags, iso_meta = self.isolation_style_scores(corp)
        flagged = iso_flags.astype(bool)
        outlier_idx = [i for i, v in enumerate(flagged.tolist()) if v]
        return {
            "count": len(examples),
            "label_buckets": dict(Counter(labels)),
            "chi_square": chi,
            "iso_meta": iso_meta,
            "isolation_candidate_indices": outlier_idx[:200],
            "heavy_distribution_anomaly_flag": iso_meta.get("isolation_fraction", 0.0)
            > self.max_iso_fraction_threshold,
        }


def filter_indices_out(examples: List[Dict[str, Any]], bad_indices: set[int]) -> List[Dict[str, Any]]:
    return [ex for i, ex in enumerate(examples) if i not in bad_indices]

