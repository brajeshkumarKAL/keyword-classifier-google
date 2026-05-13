import argparse
import difflib
import re
import string
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from main_search_terms import small_product_name


RAW_INPUT_DIR = Path("raw_input")
CLASSIFIED_DIR = Path("output/search_term_classified")
KEYWORD_CLASSIFIED_DIR = Path("output")
OUT_DIR = Path("output/search_term_outputs")
MIN_PRIMARY_POOL_SCORE = 0.45


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_single_column(path: Path, candidates: List[str]) -> List[str]:
    df = pd.read_excel(path)
    col_map = {normalize_text(c): c for c in df.columns}
    col = None
    for c in candidates:
        if c in col_map:
            col = col_map[c]
            break
    if col is None:
        for c in df.columns:
            if not df[c].dropna().astype(str).str.strip().empty:
                col = c
                break
    if col is None:
        raise ValueError(f"No usable column found in {path}")
    vals = [str(v).strip() for v in df[col].dropna().tolist()]
    return [v for v in vals if v]


def score_match(search_term: str, keyword: str) -> float:
    st = normalize_text(search_term)
    kw = normalize_text(keyword)
    if not st or not kw:
        return 0.0
    if st == kw:
        return 1.0
    st_tokens = set(st.split())
    kw_tokens = set(kw.split())
    overlap = len(st_tokens & kw_tokens) / max(len(kw_tokens), 1)
    phrase = 0.2 if f" {kw} " in f" {st} " else 0.0
    fuzzy = difflib.SequenceMatcher(None, st, kw).ratio() * 0.3
    return min(0.99, overlap + phrase + fuzzy)


def infer_trigger(search_term: str, keywords: List[str]) -> Tuple[str, float]:
    exact = [kw for kw in keywords if normalize_text(kw) == normalize_text(search_term)]
    if exact:
        return exact[0], 1.0
    scored = [(kw, score_match(search_term, kw)) for kw in keywords]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return "", 0.0
    return scored[0][0], scored[0][1]


def decide_match_type(search_term: str, keyword: str, score: float) -> str:
    st = normalize_text(search_term)
    kw = normalize_text(keyword)
    if not st or not kw:
        return "Broad Match"
    # Exact: same normalized meaning/intent proxy.
    if st == kw or score >= 0.95:
        return "Exact Match"
    # Phrase: query includes keyword meaning with modifiers.
    if f" {kw} " in f" {st} " or score >= 0.75:
        return "Phrase Match"
    # Broad: related meaning only.
    return "Broad Match"


def infer_negative_scope(search_term: str, note: str) -> str:
    st = normalize_text(search_term)
    nt = normalize_text(note)
    if "share price" in st or "stock price" in st:
        return "Account level"
    if "competitor" in nt or "cross product" in nt:
        return "Campaign level"
    return "Ad Group level"


def build_outputs(product_name: str) -> Tuple[Path, Path, Path]:
    slug = small_product_name(product_name)
    classified_path = CLASSIFIED_DIR / f"{slug}.xlsx"  # search-term classified output
    keyword_classified_path = KEYWORD_CLASSIFIED_DIR / f"{slug}.xlsx"  # main.py keyword output
    keywords_path = RAW_INPUT_DIR / f"{product_name}.xlsx"  # fallback-only universe
    if not classified_path.exists():
        raise FileNotFoundError(f"Classified search-term output not found: {classified_path}")
    if not keyword_classified_path.exists():
        raise FileNotFoundError(
            f"Keyword classified output not found: {keyword_classified_path}. "
            "Run main.py for this product first."
        )
    if not keywords_path.exists():
        raise FileNotFoundError(f"Keyword source not found: {keywords_path}")

    raw_keyword_universe = read_single_column(keywords_path, ["keyword", "keywords", "kw"])
    key_df = pd.read_excel(classified_path, sheet_name="KEYWORDS")  # search terms classified as keyword rows
    keyword_df = pd.read_excel(keyword_classified_path, sheet_name="KEYWORDS")  # source-of-truth keyword polarity pools
    required = {"Keyword", "Keyword Polarity", "Match Type", "Ad Group", "Notes"}
    missing = required - set(key_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in KEYWORDS sheet: {sorted(missing)}")
    missing_kw = required - set(keyword_df.columns)
    if missing_kw:
        raise ValueError(f"Missing required columns in keyword classified KEYWORDS sheet: {sorted(missing_kw)}")

    positive_kw_pool = (
        keyword_df[keyword_df["Keyword Polarity"].astype(str).str.strip().str.lower() == "positive"]["Keyword"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    raw_negative_kw_pool = (
        keyword_df[keyword_df["Keyword Polarity"].astype(str).str.strip().str.lower() == "negative"]["Keyword"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    # Build base-negative source while excluding structural dedup negatives.
    # This uses keyword-classification reasons and keeps only actionable negatives.
    base_negative_by_norm: Dict[str, Dict[str, str]] = {}
    structural_negative_norms = set()
    for _, row in keyword_df.iterrows():
        pol = str(row.get("Keyword Polarity", "")).strip().lower()
        if pol != "negative":
            continue
        kw_raw = str(row.get("Keyword", "")).strip()
        if not kw_raw:
            continue
        reason = str(row.get("Notes", "")).strip()
        reason_norm = reason.strip().lower()
        if reason_norm == "duplicate after normalisation":
            # Exclude structural/process artifact from final negative-keyword file.
            structural_negative_norms.add(normalize_text(kw_raw))
            continue
        k_norm = normalize_text(kw_raw)
        if not k_norm:
            continue
        # keep first actionable reason
        if k_norm not in base_negative_by_norm:
            base_negative_by_norm[k_norm] = {"keyword": kw_raw, "reason": reason}

    # Clean negative trigger pool for mapping negative search terms.
    # This avoids selecting structural negatives (duplicate-normalisation artifacts) as triggers.
    negative_kw_pool = [kw for kw in raw_negative_kw_pool if normalize_text(kw) not in structural_negative_norms]
    if not positive_kw_pool and not negative_kw_pool:
        raise ValueError("No usable classified keyword pools found in keyword output.")

    pos_rows: List[Dict[str, str]] = []
    neg_rows: List[Dict[str, str]] = []
    kw_stats: Dict[str, Dict[str, int]] = {}
    kw_scope: Dict[str, str] = {}
    kw_mapping: Dict[str, str] = {}
    mapping_stats = {
        "negative_structural_removed": len(structural_negative_norms),
        "negative_primary_pool_hits": 0,
        "negative_fallback_hits": 0,
        "low_confidence_hits": 0,
    }

    total_search_terms = 0
    for _, row in key_df.iterrows():
        st = str(row["Keyword"]).strip()
        polarity = str(row["Keyword Polarity"]).strip().lower()
        raw_match_type = str(row["Match Type"]).strip()
        ad_group = str(row["Ad Group"]).strip()
        note = str(row["Notes"]).strip()
        if not st:
            continue
        total_search_terms += 1

        # Polarity-aware trigger mapping using classified keyword pools.
        fallback_note = ""
        if polarity == "positive":
            trigger, score = infer_trigger(st, positive_kw_pool)
            if not trigger:
                trigger, score = infer_trigger(st, raw_keyword_universe)
                fallback_note = " Trigger fallback used: raw keyword universe."
            if score < 0.45:
                fallback_note += " Low trigger confidence."
        else:
            trigger, score = infer_trigger(st, negative_kw_pool)
            if trigger and score >= MIN_PRIMARY_POOL_SCORE:
                mapping_stats["negative_primary_pool_hits"] += 1
            else:
                # fallback to positive pool, as requested flow allows controlled fallback
                trigger, score = infer_trigger(st, positive_kw_pool)
                fallback_note = " Trigger fallback used: positive keyword pool."
                mapping_stats["negative_fallback_hits"] += 1
            if not trigger:
                trigger, score = infer_trigger(st, raw_keyword_universe)
                fallback_note = " Trigger fallback used: raw keyword universe."
                mapping_stats["negative_fallback_hits"] += 1
            if score < 0.45:
                fallback_note += " Low trigger confidence."
                mapping_stats["low_confidence_hits"] += 1
        match_type = decide_match_type(st, trigger, score) if trigger else (raw_match_type or "Broad Match")

        campaign = f"google_search_products_{slug}"
        mapping = f"{campaign} | {ad_group}"

        nk = normalize_text(trigger)
        if nk:
            if nk not in kw_stats:
                kw_stats[nk] = {"total": 0, "negative": 0}
                kw_mapping[nk] = mapping
            kw_stats[nk]["total"] += 1

        if polarity == "positive":
            pos_rows.append(
                {
                    "search term": st,
                    "campaign+ad group": mapping,
                    "match type": match_type,
                    "reason for being positive": f"{note}{fallback_note}".strip(),
                    "keyword triggered": trigger,
                }
            )
        else:
            scope = infer_negative_scope(st, note)
            if nk:
                kw_stats[nk]["negative"] += 1
                kw_scope[nk] = scope
            neg_rows.append(
                {
                    "search term": st,
                    "campaign+ad group": mapping,
                    "negative list": scope,
                    "reason for being negative": f"{note}{fallback_note}".strip(),
                    "KW that triggered": trigger,
                }
            )

    # Third file sources:
    # 1) Actionable base negatives from keyword-classified output (dedup-normalisation excluded).
    base_negative_keywords = {k: v["keyword"] for k, v in base_negative_by_norm.items()}

    neg_kw_rows: List[Dict[str, str]] = []
    # Build from union of base negative keywords + observed triggered keywords.
    all_candidate_norm = set(base_negative_keywords.keys()) | set(kw_stats.keys())
    norm_to_raw_fallback = {normalize_text(k): k for k in raw_keyword_universe}
    for nk in all_candidate_norm:
        kw = base_negative_keywords.get(nk) or norm_to_raw_fallback.get(nk, "")
        if not kw:
            continue
        nk = normalize_text(kw)
        stats = kw_stats.get(nk)
        total = stats["total"] if stats else 0
        negative = stats["negative"] if stats else 0
        ratio = (negative / total) if total else 0.0
        pct = round(ratio * 100, 2)
        # Corrected threshold rule: negative_for_keyword / total_search_terms > 40%
        negative_global_ratio = (negative / total_search_terms) if total_search_terms else 0.0
        negative_global_pct = round(negative_global_ratio * 100, 2)

        is_base_negative = nk in base_negative_keywords
        is_threshold_negative = negative > 0 and negative_global_ratio > 0.40
        if not is_base_negative and not is_threshold_negative:
            continue

        if is_base_negative and is_threshold_negative:
            base_reason = base_negative_by_norm.get(nk, {}).get("reason", "Classified negative keyword from keyword pipeline.")
            reason = (
                f"{base_reason} Also triggered {negative}/{total_search_terms} negative search terms "
                f"({negative_global_pct}%, >40% threshold); mapped-negative ratio {negative}/{total} ({pct}%). "
                "Tighten/remove."
            )
        elif is_base_negative:
            reason = base_negative_by_norm.get(nk, {}).get("reason", "Classified negative keyword from keyword pipeline.")
        else:
            reason = (
                f"Triggered {negative}/{total_search_terms} negative search terms "
                f"({negative_global_pct}%, >40% threshold); mapped-negative ratio {negative}/{total} ({pct}%). "
                "Broad-risk keyword added to negative list."
            )

        neg_kw_rows.append(
            {
                "keyword": kw,
                "campaign+ad group": kw_mapping.get(nk, f"google_search_products_{slug} | unknown"),
                "negative list": kw_scope.get(nk, "Ad Group level"),
                "reason for being negative": reason,
            }
        )

    if len(pos_rows) + len(neg_rows) == 0:
        raise ValueError("No classified rows found in KEYWORDS sheet.")

    # Guard: ensure no structural-negative keyword is used as trigger source in negative rows.
    for row in neg_rows:
        trig_norm = normalize_text(row.get("KW that triggered", ""))
        if trig_norm in structural_negative_norms:
            raise ValueError(
                "Structural duplicate-normalisation keyword was used as negative trigger source. "
                "Pool-cleaning invariant violated."
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pos_path = OUT_DIR / f"{slug}_positive_search_terms.xlsx"
    neg_path = OUT_DIR / f"{slug}_negative_search_terms.xlsx"
    neg_kw_path = OUT_DIR / f"{slug}_negative_keywords.xlsx"

    pd.DataFrame(
        pos_rows,
        columns=[
            "search term",
            "campaign+ad group",
            "match type",
            "reason for being positive",
            "keyword triggered",
        ],
    ).to_excel(pos_path, index=False)
    pd.DataFrame(
        neg_rows,
        columns=[
            "search term",
            "campaign+ad group",
            "negative list",
            "reason for being negative",
            "KW that triggered",
        ],
    ).to_excel(neg_path, index=False)
    pd.DataFrame(
        neg_kw_rows,
        columns=[
            "keyword",
            "campaign+ad group",
            "negative list",
            "reason for being negative",
        ],
    ).to_excel(neg_kw_path, index=False)
    print(
        "[Build Summary] "
        f"structural negatives removed from trigger pool={mapping_stats['negative_structural_removed']}, "
        f"negative primary-pool hits={mapping_stats['negative_primary_pool_hits']}, "
        f"negative fallbacks={mapping_stats['negative_fallback_hits']}, "
        f"low-confidence mappings={mapping_stats['low_confidence_hits']}"
    )
    return pos_path, neg_path, neg_kw_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 3 manager-format outputs from main_search_terms classification.")
    parser.add_argument("--product", required=True, help="Product name.")
    args = parser.parse_args()
    p1, p2, p3 = build_outputs(args.product)
    print(f"Positive search-term file: {p1}")
    print(f"Negative search-term file: {p2}")
    print(f"Negative keyword file: {p3}")


if __name__ == "__main__":
    main()
