import argparse
import concurrent.futures
import difflib
import json
import os
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

INPUT_KEYWORDS_CSV = Path("input/input_keywords.csv")
POSITIONING_XLSX = Path("input/Test_Positioning document.xlsx")
OUTPUT_DIR = Path("output")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-5-mini"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


TRANSACTIONAL_MODIFIERS = {"buy", "price", "online", "order", "shop", "purchase"}
EVALUATION_MODIFIERS = {"best", "top", "review", "reviews", "compare", "comparison", "vs", "which"}
HARD_INFORMATIONAL_NEGATIVE_TERMS = {"review", "reviews"}
INFORMATIONAL_TERMS = {
    "benefits",
    "uses",
    "side effects",
    "side effect",
    "dosage",
    "ingredients",
    "how to",
    "what is",
    "meaning",
    "wikipedia",
    "explain",
    "learn",
    "understand",
    "definition",
    "symptoms",
    "symptom",
    "causes",
    "cause",
    "prevention",
    "diagnosis",
    "risk factors",
    "complications",
    "types",
}
MARKETPLACE_TERMS = {
    "amazon", "flipkart", "1mg", "netmeds", "meesho", "bigbasket", "zepto", 
    "instamart", "blinkit", "swiggy", "zomato", "myntra", "nykaa", "purplle", 
    "healthkart", "pharmeasy", "apollo pharmacy", "medplus", "practo", "tata 1mg"
}
NON_PURCHASE_TERMS = {
    "doctor", "clinic", "near me", "exercise", "yoga", "home remedy", "home remedies", 
    "diet", "food", "recipe", "nutrition", "hospital", 
    "surgery", "operation", "consultation", "specialist", "therapist", 
    "therapy", "massage", "spa", "salon", "diy", "make at home", "workout", 
    "gym", "fitness", "stretching", "physiotherapy", "treatment at home", 
    "naturopathy"
}
FORMAT_TERMS = {
    "oil", "tablet", "syrup", "tonic", "capsule", "choorna", "churna", 
    "keram", "tailam", "thailam", "asava", "arishtam", "lehyam", "bhasma", "vati", "gutika", 
    "kashayam", "kwath", "powder", "gel", "cream", "soap", "ointment", "spray", 
    "drop", "drops", "juice"
}
FORMAT_CANONICAL_MAP = {
    "keram": "oil",
    "tailam": "oil",
    "thailam": "oil",
    "oil": "oil",
    "choorna": "powder",
    "churna": "powder",
}
SOLUTION_TERMS = {
    "medicine", "remedies", "remedy", "treatment", "solution", "care", 
    "herbs", "ayurveda", "ayurvedic"
}
COMPETITOR_FALLBACK_TERMS = {
    "dabur", "baidyanath", "patanjali", "himalaya", "zandu", "kottakkal", 
    "arya vaidya sala", "jiva", "kapiva", "banyan botanicals", "maharishi ayurveda", 
    "kama ayurveda", "forest essentials", "biotique", "vicco", "charak", 
    "sandu", "dhootapapeshwar", "vaidyaratnam", "sri sri tattva"
}
USE_CASE_STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "that",
    "this",
    "helps",
    "help",
    "relief",
    "quick",
    "easy",
    "pain",
    "joint",
    "muscle",
    "general",
    "during",
    "after",
    "being",
    "reduce",
    "improve",
    "providing",
    "persistent",
    "continue",
    "episodes",
}

USE_CASE_LIGHT_STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "that",
    "this",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "by",
    "or",
}

SEVERITY_QUALITY_FILLER = {
    "severe",
    "chronic",
    "acute",
    "mild",
    "extreme",
    "high",
    "excessive",
    "best",
    "good",
    "top",
    "effective",
    "powerful",
    "strong",
    "natural",
    "problem",
    "issue",
    "trouble",
    "disorder",
    "condition",
    "related",
    "type",
    "and",
    "with",
    "due to",
}


@dataclass
class ProductContext:
    product_name: str
    product_slug: str
    product_url: str
    primary_use_case: str
    use_case_and_benefits: str
    key_benefits: str
    use_case_text: str
    competitors: List[str]
    allowed_format: str
    all_product_names: List[str]
    product_aliases: List[str]


@dataclass
class LLMConfig:
    base_url: str
    model: str
    api_key_env: str


@dataclass
class KeywordRecord:
    keyword: str
    fingerprint: str
    keyword_polarity: str
    match_type: str
    intent_type: str
    funnel_stage: str
    ad_group: str
    notes: str
    english_equivalent: str = ""
    english_equivalent_canonical: str = ""
    original_language: str = "english"


HINGLISH_TOKENS = {
    "dawa", "ki", "ke", "ka", "ko", "se", "mein", "me",
    "madhumeh", "madhumeh", "madhumeha",
    "lakshan", "lakshad", "lakshana", "lakshan",
    "gharelu", "gharelunuskhe", "nuskhe", "nuskha",
    "upay", "upchar", "ilaj", "ilaaj",
    "desi", "ghar", "baba", "ramdev",
    "pet", "petdard", "petdard", "petdard",
    "sardi", "khansi", "khansi", "zukam",
    "jukaam", "bukhar", "bukhaar",
    "pairo", "pairon", "pairo", "pairo",
    "dard", "dardon", "dard",
    "takleef", "takleefon", "takleef",
    "shakkar", "shugar", "shugger",
    "cheeni", "chini",
    "khoon", "khoon",
    "dil", "dilki",
    "jigar", "jigarki",
    "pachan", "pachan",
    "swasthya", "swasth",
    "rog", "rogon", "rog",
    "vyadhi", "vyadhiyan",
    "aushadhi", "aushadh",
    "jadi", "jadi", "buti", "butiyan",
    "vajan", "vajan", "motapa", "patla",
    "twacha", "twacha", "rang", "rangat",
    "baal", "baalon", "jhadna", "jhad",
    "neend", "neend", "nind", "nind",
    "yaadash", "yaad", "bhoolna",
    "thakan", "thakavat", "kamzori",
    "shakti", "shakti", "bal", "bal",
    "virya", "virya", "shukranu",
    "garbhpat", "garbhpat",
    "masik", "masikdharm", "mahavari",
    "stan", "stan", "stan",
    "garbhashay", "garbhashay",
    "pitt", "pitt", "pitt",
    "vata", "vata", "kapha",
    "agni", "agni", "jatharagni",
    "aam", "aam", "aam",
    "gola", "goliya", "goli",
    "kadha", "kadha", "kadha",
    "ark", "ark", "ark",
    "ras", "ras", "ras",
    "pak", "pak", "pak",
    "avaleh", "avaleh", "avaleh",
    "churan", "churan", "churan",
    "tel", "tel", "tail",
    "malish", "malish", "malish",
    "swedan", "swedan", "swedan",
    "patti", "patti", "lep",
    "dhup", "dhup", "dhupan",
}


def detect_language(keyword: str) -> str:
    """Fast heuristic to detect original language of a keyword."""
    kw = normalize_keyword(keyword)
    # Devanagari script = Hindi
    if re.search(r"[\u0900-\u097F]", kw):
        return "hindi"
    # Known Hinglish tokens
    tokens = set(kw.split())
    if tokens & HINGLISH_TOKENS:
        return "hinglish"
    return "english"


def translate_keywords_batch(
    records: List[KeywordRecord], llm_client: OpenAI, llm_model: str
) -> List[KeywordRecord]:
    """
    Detect original_language and provide english_equivalent for non-English keywords.
    Uses LLM only for hinglish/hindi keywords; english keywords are fast-pathed.
    """
    # Fast-path: separate english from non-english
    english_records: List[KeywordRecord] = []
    to_translate: List[KeywordRecord] = []
    for r in records:
        lang = detect_language(r.keyword)
        r.original_language = lang
        if lang == "english":
            r.english_equivalent = r.keyword
            english_records.append(r)
        else:
            to_translate.append(r)

    if not to_translate:
        return records

    batch_size = 10
    total_batches = (len(to_translate) + batch_size - 1) // batch_size

    system_prompt = (
        "You are a multilingual keyword translator for Google Ads.\n\n"
        "Task: For each keyword, detect its original_language and provide its english_equivalent.\n\n"
        "Language definitions:\n"
        "- english: Native English words only (e.g., 'diabetes medicine', 'buy ayurvedic oil').\n"
        "- hinglish: Hindi words written in Latin/English script (e.g., 'madhumeh ki dawa', 'sugar ke lakshan', 'pairo ka dard').\n"
        "- hindi: Devanagari script (e.g., 'मधुमेह की दवा', 'शुगर के लक्षण').\n\n"
        "Translation rules:\n"
        "1. For Hinglish: translate each Hindi token to English and rephrase naturally.\n"
        "   Examples:\n"
        "   - 'madhumeh ki ayurvedic dawa' → 'ayurvedic medicine for diabetes'\n"
        "   - 'sugar ke lakshan' → 'symptoms of diabetes'\n"
        "   - 'pairo ka dard ki dawa' → 'medicine for leg pain'\n"
        "   - 'sugar ki ayurvedic dawa' → 'ayurvedic medicine for sugar'\n"
        "   - 'pet dard ka ilaj' → 'treatment for stomach pain'\n"
        "   - 'baal jhadne ka upay' → 'remedy for hair fall'\n"
        "   - 'vajan kam karne ki dawa' → 'medicine for weight loss'\n"
        "2. For Hindi (Devanagari): translate to natural English search query.\n"
        "   Examples:\n"
        "   - 'मधुमेह की आयुर्वेदिक दवा' → 'ayurvedic medicine for diabetes'\n"
        "   - 'शुगर के लक्षण' → 'symptoms of diabetes'\n"
        "   - 'पैरों का दर्द की दवा' → 'medicine for leg pain'\n"
        "3. Preserve brand names (Kerala Ayurveda, Dabur, Patanjali) and product names as-is.\n"
        "4. Preserve product formats (oil, tablet, churna, syrup) in English.\n"
        "5. Output english_equivalent should read like a natural English Google search query.\n\n"
        "Return strict JSON only with key 'results'.\n"
        "Each item must be: keyword, original_language (english|hinglish|hindi), english_equivalent.\n"
        "Return exactly one item for every input keyword."
    )

    def translate_batch(batch: List[KeywordRecord], batch_num: int) -> None:
        keywords = [r.keyword for r in batch]
        user_prompt = "Keywords:\n- " + "\n- ".join(keywords)
        data = chat_json(llm_client, llm_model, system_prompt, user_prompt)
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ValueError("LLM translation response missing required 'results' list")

        by_kw: Dict[str, dict] = {}
        for item in data["results"]:
            if not isinstance(item, dict):
                continue
            kw = normalize_keyword(str(item.get("keyword", "")))
            if kw:
                by_kw[kw] = item

        missing = [r for r in batch if normalize_keyword(r.keyword) not in by_kw]
        if missing:
            print(f"[Progress] Translation retrying {len(missing)} missing keywords in batch {batch_num}")
            retry_keywords = [r.keyword for r in missing]
            retry_prompt = "Keywords:\n- " + "\n- ".join(retry_keywords)
            retry_data = chat_json(llm_client, llm_model, system_prompt, retry_prompt)
            if isinstance(retry_data, dict) and isinstance(retry_data.get("results"), list):
                for item in retry_data["results"]:
                    if isinstance(item, dict):
                        kw = normalize_keyword(str(item.get("keyword", "")))
                        if kw:
                            by_kw[kw] = item

        for r in batch:
            kw_norm = normalize_keyword(r.keyword)
            item = by_kw.get(kw_norm, {})
            lang = str(item.get("original_language", r.original_language)).strip().lower()
            equiv = str(item.get("english_equivalent", r.keyword)).strip().lower()
            if lang in {"english", "hinglish", "hindi"}:
                r.original_language = lang
            if equiv:
                r.english_equivalent = equiv
            else:
                r.english_equivalent = r.keyword

    for i in range(0, len(to_translate), batch_size):
        batch = to_translate[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"[Progress] Translation batch {batch_num}/{total_batches} ({len(batch)} keywords)")
        translate_batch(batch, batch_num)

    return records


def normalize_keyword(keyword: str) -> str:
    if not isinstance(keyword, str):
        keyword = "" if pd.isna(keyword) else str(keyword)
    text = keyword.lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonicalize_english_equivalents_batch(
    records: List[KeywordRecord], llm_client: OpenAI, llm_model: str
) -> List[KeywordRecord]:
    """
    LLM-based semantic canonicalization for english_equivalent.
    Produces a stable canonical phrase so synonymic variants map together.
    """
    if not records:
        return records

    batch_size = 20
    system_prompt = (
        "You canonicalize English search queries for semantic grouping in Google Ads.\n"
        "Task: for each input keyword phrase, return english_equivalent_canonical.\n"
        "Rules:\n"
        "1. Keep meaning; do not change intent.\n"
        "2. Merge synonymic wording to one natural canonical phrase.\n"
        "3. Normalize minor wording variants (dawa/dawai/dava style meaning-equivalent phrases should map together).\n"
        "4. Keep brand names and condition terms intact.\n"
        "5. Output concise lowercase canonical English phrase.\n"
        "Return strict JSON: {\"results\":[{\"keyword\":\"...\",\"english_equivalent_canonical\":\"...\"}]}\n"
        "Return exactly one result per input."
    )

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        user_prompt = "Keywords:\n- " + "\n- ".join([r.english_equivalent or r.keyword for r in batch])
        data = chat_json(llm_client, llm_model, system_prompt, user_prompt)
        by_kw: Dict[str, str] = {}
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            for item in data["results"]:
                if not isinstance(item, dict):
                    continue
                k = normalize_keyword(str(item.get("keyword", "")))
                c = normalize_keyword(str(item.get("english_equivalent_canonical", "")))
                if k and c:
                    by_kw[k] = c
        for r in batch:
            source = normalize_keyword(r.english_equivalent or r.keyword)
            r.english_equivalent_canonical = by_kw.get(source, source)

    return records


def slugify(value: str) -> str:
    return re.sub(r"\s+", "_", normalize_keyword(value))


def canonical_product_name(value: str) -> str:
    t = normalize_keyword(value)
    t = re.sub(r"\(.*?\)", " ", t)
    t = re.sub(
        r"\b(oil|thailam|tailam|asava|arishtam|arishta|tablet|tablets|capsule|capsules|syrup|keram|choornam|choorna|churna|lehyam|cream|gel|drink)\b",
        " ",
        t,
    )
    t = re.sub(r"\s+", " ", t).strip()
    return t


def small_product_name(value: str) -> str:
    canon = canonical_product_name(value)
    return re.sub(r"\s+", "_", canon).strip("_")


def infer_product_format(product_name: str, use_case_text: str) -> str:
    joined = f"{product_name} {use_case_text}".lower()
    for fmt in FORMAT_TERMS:
        if fmt in joined:
            return FORMAT_CANONICAL_MAP.get(fmt, fmt)
    return "medicine"


def product_family_aliases(product_name: str, allowed_format: str) -> List[str]:
    base = normalize_keyword(product_name)
    if not base:
        return []

    aliases = {base, canonical_product_name(product_name)}
    tokens = [t for t in base.split() if t]
    if tokens:
        aliases.add(" ".join(tokens))

    # Build format-normalized family variants.
    format_like = {"oil", "keram", "tailam", "thailam"}
    core_tokens = [t for t in tokens if t not in format_like]
    if core_tokens:
        core_phrase = " ".join(core_tokens).strip()
        if core_phrase:
            aliases.add(core_phrase)
            aliases.add(f"{core_phrase} oil")
            aliases.add(f"{core_phrase} keram")
            aliases.add(f"{core_phrase} thailam")
            aliases.add(f"{core_phrase} tailam")

    # Orthographic variants for common transliteration drift.
    expanded = set(aliases)
    for a in list(aliases):
        if "bringadi" in a:
            expanded.add(a.replace("bringadi", "bhringadi"))
        if "bhringadi" in a:
            expanded.add(a.replace("bhringadi", "bringadi"))

    # Keep normalized, non-empty unique aliases only.
    cleaned = []
    seen = set()
    for a in expanded:
        n = normalize_keyword(a)
        if not n or n in seen:
            continue
        seen.add(n)
        cleaned.append(n)
    return cleaned


def has_product_alias_match(keyword: str, context: ProductContext) -> bool:
    kw = normalize_keyword(keyword)
    if not kw:
        return False
    kw_pad = f" {kw} "
    for alias in context.product_aliases:
        a = normalize_keyword(alias)
        if not a:
            continue
        if f" {a} " in kw_pad:
            return True
    return False


def has_probable_brand_context(keyword: str, context: ProductContext) -> bool:
    kw = normalize_keyword(keyword)
    if "kerala ayurveda" in kw:
        return True
    return "kerala" in kw and has_product_alias_match(kw, context)



def parse_competitors(raw: str) -> List[str]:
    if not isinstance(raw, str):
        return []
    parts = re.split(r"\n|,|→|->|\|", raw)
    brands = []
    for p in parts:
        t = normalize_keyword(p)
        if not t:
            continue
        if len(t) > 2 and t not in brands:
            brands.append(t)
    return brands


def clean_cell_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def build_llm_client(config: LLMConfig) -> OpenAI:
    load_dotenv()
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise ValueError(f"Missing required API key in environment variable '{config.api_key_env}'")
    return OpenAI(base_url=config.base_url, api_key=api_key)


def chat_json(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON response: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc


def load_positioning_context(positioning_path: Path, product_name: str) -> ProductContext:
    df = pd.read_excel(positioning_path, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    if "Unnamed: 0" in df.columns and str(df.iloc[0, 0]).lower().strip() == "product name":
        df.columns = [str(v).strip() for v in df.iloc[0].tolist()]
        df = df.iloc[1:].reset_index(drop=True)

    col_map = {normalize_keyword(c): c for c in df.columns}
    product_col = col_map.get("product name")
    url_col = col_map.get("website links") or col_map.get("website links ")
    primary_use_case_col = col_map.get("primary use case")
    use_case_and_benefits_col = col_map.get("use case and benefits")
    key_benefits_col = col_map.get("key benefits")
    competitor_col = col_map.get("competitor brands")

    if not product_col or not url_col:
        raise ValueError("Positioning document missing required columns: Product Name / Website links")

    target_raw = product_name.strip().lower()
    target_canon = canonical_product_name(product_name)
    match = df[df[product_col].astype(str).str.strip().str.lower() == target_raw]
    if match.empty:
        canon_series = df[product_col].astype(str).apply(canonical_product_name)
        match = df[canon_series == target_canon]
    if match.empty:
        raise ValueError(f"Product '{product_name}' not found in positioning document")

    row = match.iloc[0]
    pname = clean_cell_text(row[product_col])
    purl = clean_cell_text(row[url_col])
    primary_use_case = clean_cell_text(row[primary_use_case_col]) if primary_use_case_col else ""
    use_case_and_benefits = clean_cell_text(row[use_case_and_benefits_col]) if use_case_and_benefits_col else ""
    key_benefits = clean_cell_text(row[key_benefits_col]) if key_benefits_col else ""
    use_case = " ".join(
        part for part in [primary_use_case, use_case_and_benefits, key_benefits] if part
    ).strip()
    competitors = parse_competitors(row[competitor_col]) if competitor_col else []

    all_products = [normalize_keyword(str(v)) for v in df[product_col].dropna().astype(str).tolist()]
    allowed_format = infer_product_format(pname, use_case)
    return ProductContext(
        product_name=pname,
        product_slug=slugify(pname),
        product_url=purl,
        primary_use_case=normalize_keyword(primary_use_case),
        use_case_and_benefits=normalize_keyword(use_case_and_benefits),
        key_benefits=normalize_keyword(key_benefits),
        use_case_text=normalize_keyword(use_case),
        competitors=competitors,
        allowed_format=allowed_format,
        all_product_names=all_products,
        product_aliases=product_family_aliases(pname, allowed_format),
    )


def load_product_keywords(input_path: Path, product_name: str) -> pd.DataFrame:
    if not input_path.exists():
        raise ValueError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    col_map = {normalize_keyword(str(c)): c for c in df.columns}
    product_col = col_map.get("product_name") or col_map.get("product name")
    keyword_col = col_map.get("keyword")
    if not product_col or not keyword_col:
        raise ValueError("Input CSV must contain required columns: product_name, keyword")

    target = product_name.strip().lower()
    target_canon = canonical_product_name(product_name)
    filtered = df[df[product_col].astype(str).str.strip().str.lower() == target]
    if filtered.empty:
        canon_series = df[product_col].astype(str).apply(canonical_product_name)
        filtered = df[canon_series == target_canon]
    if filtered.empty:
        raise ValueError(f"No keywords found in {input_path} for product '{product_name}'")

    out = filtered[[keyword_col]].rename(columns={keyword_col: "Keyword"})
    return out


def step_0_startup_checks(context: ProductContext) -> None:
    if not context.product_url.startswith("http"):
        raise ValueError("Product URL missing or invalid in positioning document")


def step_1_keyword_cleaning(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    out = df.copy()
    out["keyword"] = out["Keyword"].apply(normalize_keyword)
    out = out[out["keyword"] != ""]

    # Identify only the duplicate rows that will actually be dropped (keep first as canonical)
    duplicates_to_drop = out[out.duplicated(subset=["keyword"], keep="first")]
    duplicate_negatives: List[Dict[str, str]] = []
    if not duplicates_to_drop.empty:
        for _, row in duplicates_to_drop.iterrows():
            duplicate_negatives.append({
                "Keyword": row["keyword"],
                "Fingerprint": "",
                "Keyword Polarity": "Negative",
                "Intent Type": "Informational",
                "Funnel Stage": "TOF",
                "Reason for Exclusion": "Duplicate after normalization",
            })

    out = out.drop_duplicates(subset=["keyword"]).reset_index(drop=True)
    return out, duplicate_negatives


def step_1b_fingerprint_semantic_dedup(
    records: List[KeywordRecord], context: ProductContext, llm_client: OpenAI, llm_model: str
) -> List[KeywordRecord]:
    keywords = [r.keyword for r in records]
    if not keywords:
        return records
    batch_size = 10
    total_batches = (len(keywords) + batch_size - 1) // batch_size

    system_prompt = (
        "You are an expert Google Ads keyword grouping assistant.\n\n"
        "Goal:\n"
        "Create an ad-group-level fingerprint for each keyword, then group keywords with the same fingerprint.\n\n"
        "Context:\n"
        "- Brand: Kerala Ayurveda\n"
        "- Market: India\n"
        f"- Product: {context.product_name}\n"
        f"- Product website: {context.product_url}\n"
        f"- Competitors: {', '.join(sorted(competitor_terms(context)))}\n"
        f"- Product format: {context.allowed_format}\n"
        f"- Product use cases: {context.use_case_text}\n\n"
        "Ad-group fingerprint means:\n"
        "A compact representation of whether keywords can share the same ad group, ad copy, landing page, and bid strategy.\n\n"
        "For each keyword, create a fingerprint object with the following fields and meanings:\n\n"
        "1. brand_scope (brand | competitor | generic)\n"
        "   Meaning: tells whether the keyword is about Kerala Ayurveda brand terms, competitor terms, or non-brand generic demand.\n"
        "2. ad_group_intent (product | brand_product | condition | evaluation | competitor | informational)\n"
        "   Meaning: coarse campaign intent bucket for grouping keywords into ad-group strategy type.\n"
        "3. match_intent_core (string)\n"
        "   Meaning: normalized core query meaning (for example, a compact semantic theme like hair_fall_solution).\n"
        "   IMPORTANT: Derive this from the english_equivalent field for consistency across languages.\n"
        "4. query_shape (product_only | product_plus_condition | condition_solution | brand_product | competitor_product | informational | commercial_modifier)\n"
        "   Meaning: structural pattern of the query (what combination of concepts appears).\n"
        "5. product_family (string | null)\n"
        "   Meaning: normalized product line or solution category receiving traffic.\n"
        "6. product_variant (string | null)\n"
        "   Meaning: specific variant signals such as size/count/version. Should not duplicate format info.\n"
        "7. product_format (oil | powder | decoction | fermented_liquid | tablet | capsule | syrup | gel | other | unknown)\n"
        "   Meaning: dosage/form-factor classification used for relevance and format alignment.\n"
        "8. condition_or_need (string | null)\n"
        "   Meaning: problem/need-state being solved (for example, pain, cough, hair fall).\n"
        "   IMPORTANT: Derive this from the english_equivalent field for consistency across languages.\n"
        "9. audience (string | null)\n"
        "   Meaning: audience qualifier when present (for example, kids/women/general).\n"
        "10. commercial_intent (buy | price | best | review | comparison | generic | informational)\n"
        "    Meaning: buying-stage signal extracted from modifiers and phrasing.\n"
        "11. landing_page_type (product_detail | category | condition_page | brand_page | comparison_page | educational_page)\n"
        "    Meaning: intended destination page class for this keyword.\n"
        "12. ad_copy_angle (buy_now | price_value | efficacy_benefit | brand_trust | comparison | education | review_social_proof)\n"
        "    Meaning: primary message angle expected to perform best for the keyword group.\n"
        "13. phrase_match_expansion_risk (low | medium | high)\n"
        "    Meaning: risk level that phrase match may expand into unwanted traffic.\n"
        "14. negative_keyword_needs (array of strings)\n"
        "    Meaning: suggested negatives that should protect budget from irrelevant matches.\n"
        "15. risk_flags (object with booleans: medical_claim_sensitive, disease_term_present, competitor_term_present, ambiguous_product_mapping)\n"
        "    Meaning: policy/compliance and ambiguity indicators useful for review and control.\n"
        "16. english_equivalent (string)\n"
        "    Meaning: The English translation/normalization of the keyword.\n"
        "    For English keywords, use the keyword itself.\n"
        "    For Hinglish/Hindi keywords, use the pre-translated equivalent provided in the input.\n"
        "    This ensures 'madhumeh ki dawa' and 'diabetes medicine' share the same semantic fingerprint.\n"
        "17. original_language (english | hinglish | hindi)\n"
        "    Meaning: The original script/language the user searched in.\n"
        "    Hinglish = Hindi words written in Latin script (e.g., 'dawa', 'ke lakshan').\n"
        "    Hindi = Devanagari script (e.g., 'मधुमेह').\n\n"
        "Canonicalize synonyms:\n"
        "- buy, order, shop, online → buy\n"
        "- price, cost, rate, MRP → price\n"
        "- best, top → best\n"
        "- review, reviews → review\n"
        "- taila, tailam, thailam → oil\n"
        "- churna, choornam, churnam → powder\n"
        "- kashayam, kwath → decoction\n"
        "- arishta, arishtam, asava → fermented_liquid\n\n"
        "Grouping rules (Core vs Soft Fields):\n"
        "1. CORE FIELDS (Must Match 100%): You may only group keywords if they exactly share: brand_scope, ad_group_intent, match_intent_core, product_family, product_format, condition_or_need, commercial_intent, landing_page_type, ad_copy_angle, english_equivalent, and original_language.\n"
        "   IMPORTANT: english_equivalent must be treated semantically (same meaning/synonymic phrasing should be considered a match), not strict word-by-word text.\n"
        "2. SOFT FIELDS (Can Differ): Keywords can be safely grouped even if their query_shape, audience, phrase_match_expansion_risk, negative_keyword_needs, or risk_flags differ.\n"
        "3. RESOLVING SOFT FIELDS FOR THE GROUP FINGERPRINT:\n"
        "   - negative_keyword_needs: Combine all necessary negatives from all keywords in the group.\n"
        "   - phrase_match_expansion_risk: Take the highest risk level present.\n"
        "   - risk_flags: If ANY keyword in the group has a flag set to true, set it to true for the group.\n"
        "   - query_shape / audience: Choose the value that best represents the primary intent of the group.\n\n"
        "Return JSON only with this structure:\n"
        "{\n"
        '  "groups": [\n'
        "    {\n"
        '      "group_id": "adg_001",\n'
        '      "ad_group_label": "string",\n'
        '      "fingerprint": {\n'
        '        "brand_scope": "generic",\n'
        '        "ad_group_intent": "condition",\n'
        '        "match_intent_core": "hair_fall_solution",\n'
        '        "query_shape": "condition_solution",\n'
        '        "product_family": "ayurvedic_hair_oil",\n'
        '        "product_variant": null,\n'
        '        "product_format": "oil",\n'
        '        "condition_or_need": "hair_fall",\n'
        '        "audience": null,\n'
        '        "commercial_intent": "buy",\n'
        '        "landing_page_type": "condition_page",\n'
        '        "ad_copy_angle": "efficacy_benefit",\n'
        '        "phrase_match_expansion_risk": "medium",\n'
        '        "negative_keyword_needs": ["free", "home remedy"],\n'
        '        "risk_flags": {\n'
        '          "medical_claim_sensitive": false,\n'
        '          "disease_term_present": false,\n'
        '          "competitor_term_present": false,\n'
        '          "ambiguous_product_mapping": false\n'
        "        },\n"
        '        "english_equivalent": "ayurvedic medicine for hair fall",\n'
        '        "original_language": "english"\n'
        "      },\n"
        '      "keywords": [\n'
        '        "keyword 1",\n'
        '        "keyword 2"\n'
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )
    fingerprint_map: Dict[str, str] = {}
    kw_meta: Dict[str, KeywordRecord] = {r.keyword: r for r in records}

    def collect_fingerprint_map(batch_keywords: List[str], batch_num: int) -> Dict[str, str]:
        # Build enriched prompt with english_equivalent and original_language
        lines: List[str] = []
        for kw in batch_keywords:
            rec = kw_meta.get(kw)
            if rec and rec.original_language != "english":
                lines.append(f"- {kw}  [lang={rec.original_language}, en=\"{rec.english_equivalent}\"]")
            else:
                lines.append(f"- {kw}")
        user_prompt = "Keywords:\n" + "\n".join(lines)
        data = chat_json(llm_client, llm_model, system_prompt, user_prompt)
        if not isinstance(data, dict) or not isinstance(data.get("groups"), list):
            raise ValueError("LLM fingerprint response missing required 'groups' list")

        input_set = set(batch_keywords)
        seen: set = set()
        local_fingerprint_map: Dict[str, str] = {}
        for group in data["groups"]:
            if not isinstance(group, dict):
                continue
            members_raw = group.get("keywords", [])
            if not isinstance(members_raw, list):
                continue
            members = [normalize_keyword(str(k)) for k in members_raw]
            members = [k for k in members if k in input_set]
            if not members:
                continue
            fp_obj = group.get("fingerprint", {})
            if isinstance(fp_obj, dict):
                fp_obj = dict(fp_obj)
                rec0 = kw_meta.get(members[0]) if members else None
                canonical_en = ""
                if rec0:
                    canonical_en = normalize_keyword(rec0.english_equivalent_canonical or rec0.english_equivalent or rec0.keyword)
                fp_obj["english_equivalent"] = canonical_en
                fp_obj["original_language"] = "normalized"
            fp_str = json.dumps(fp_obj, sort_keys=True, ensure_ascii=True) if isinstance(fp_obj, dict) else ""
            if fp_str:
                for m in members:
                    local_fingerprint_map[m] = fp_str
            for m in members:
                seen.add(m)

        missing = [kw for kw in batch_keywords if kw not in seen]
        if missing:
            # Self-heal with per-keyword LLM calls for any uncovered keywords.
            for kw in missing:
                rec = kw_meta.get(kw)
                if rec and rec.original_language != "english":
                    single_prompt = f"Keywords:\n- {kw}  [lang={rec.original_language}, en=\"{rec.english_equivalent}\"]"
                else:
                    single_prompt = "Keywords:\n- " + kw
                single_data = chat_json(llm_client, llm_model, system_prompt, single_prompt)
                if not isinstance(single_data, dict) or not isinstance(single_data.get("groups"), list):
                    raise ValueError(f"LLM fingerprint grouping missing keyword in batch {batch_num}: '{kw}'")
                covered = False
                for group in single_data["groups"]:
                    if not isinstance(group, dict):
                        continue
                    members_raw = group.get("keywords", [])
                    if not isinstance(members_raw, list):
                        continue
                    members = [normalize_keyword(str(k)) for k in members_raw]
                    if kw in members:
                        fp_obj = group.get("fingerprint", {})
                        if isinstance(fp_obj, dict):
                            fp_obj = dict(fp_obj)
                            canonical_en = normalize_keyword(rec.english_equivalent_canonical or rec.english_equivalent or rec.keyword)
                            fp_obj["english_equivalent"] = canonical_en
                            fp_obj["original_language"] = "normalized"
                        fp_str = json.dumps(fp_obj, sort_keys=True, ensure_ascii=True) if isinstance(fp_obj, dict) else ""
                        if fp_str:
                            local_fingerprint_map[kw] = fp_str
                        covered = True
                        break
                if not covered:
                    raise ValueError(f"LLM fingerprint grouping missing keyword in batch {batch_num}: '{kw}'")
        return local_fingerprint_map

    batches: List[Tuple[int, List[str]]] = []
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i : i + batch_size]
        batch_num = i // batch_size + 1
        batches.append((batch_num, batch))

    def process_batch(item: Tuple[int, List[str]]) -> Tuple[int, Dict[str, str]]:
        batch_num, batch = item
        print(f"[Progress] Step 4/9 Fingerprint dedup: batch {batch_num}/{total_batches} ({len(batch)} keywords)")
        try:
            batch_fp_map = collect_fingerprint_map(batch, batch_num)
        except Exception:
            print(f"[Progress] Step 4/9 Fingerprint dedup: retrying batch {batch_num}")
            batch_fp_map = collect_fingerprint_map(batch, batch_num)
        return batch_num, batch_fp_map

    max_workers = min(6, total_batches) if total_batches > 0 else 1
    ordered_fp_maps: Dict[int, Dict[str, str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, b) for b in batches]
        for fut in concurrent.futures.as_completed(futures):
            batch_num, batch_fp_map = fut.result()
            ordered_fp_maps[batch_num] = batch_fp_map

    for batch_num in range(1, total_batches + 1):
        fingerprint_map.update(ordered_fp_maps.get(batch_num, {}))

    missing_fp = [kw for kw in keywords if kw not in fingerprint_map]
    if missing_fp:
        raise ValueError(f"LLM fingerprint mapping missing {len(missing_fp)} keywords")

    for rec in records:
        rec.fingerprint = fingerprint_map.get(rec.keyword, "")
    return records


def contains_any(text: str, terms: set) -> bool:
    return any(term in text for term in terms)


def competitor_terms(context: ProductContext) -> set:
    return set(context.competitors) | COMPETITOR_FALLBACK_TERMS


def contains_term_phrase(text: str, term: str) -> bool:
    text_norm = f" {normalize_keyword(text)} "
    term_norm = normalize_keyword(term)
    if not term_norm:
        return False
    return f" {term_norm} " in text_norm


def has_competitor_term(keyword: str, context: ProductContext) -> bool:
    return any(contains_term_phrase(keyword, c) for c in competitor_terms(context))


def has_product_reference(keyword: str, context: ProductContext) -> bool:
    if has_product_alias_match(keyword, context):
        return True
    own = normalize_keyword(context.product_name)
    if own in keyword:
        return True
    stop = FORMAT_TERMS
    core_tokens = [t for t in own.split() if len(t) >= 4 and t not in stop]
    if any(t in keyword for t in core_tokens):
        return True
    kw_tokens = keyword.split()
    for p in core_tokens:
        if len(p) < 6:
            continue
        for k in kw_tokens:
            if difflib.SequenceMatcher(None, p, k).ratio() >= 0.86:
                return True
    return False


def canonicalize_format_token(token: str) -> str:
    return FORMAT_CANONICAL_MAP.get(token, token)


def is_competitor_product_keyword(keyword: str, context: ProductContext) -> bool:
    return has_competitor_term(keyword, context) and has_product_reference(keyword, context)


def extract_use_case_terms(context: ProductContext) -> set:
    source = " ".join(
        [
            context.primary_use_case or "",
            context.use_case_and_benefits or "",
            context.key_benefits or "",
        ]
    )
    source_norm = normalize_keyword(source)
    source_tokens = source_norm.split()
    product_tokens = set(normalize_keyword(context.product_name).split())
    blocked = USE_CASE_STOPWORDS | product_tokens | FORMAT_TERMS | {
        "kerala",
        "ayurveda",
        "ayurvedic",
        "medicine",
        "medicines",
        "thailam",
        "choornam"
    }

    # Token-level signals (keep meaningful condition/benefit words).
    terms = {t for t in source_tokens if len(t) >= 3 and t not in blocked}

    # Phrase-level signals from adjacent tokens to capture intent like "joint pain", "pain relief".
    phrase_terms: set = set()
    filtered_for_phrase = [
        t for t in source_tokens
        if len(t) >= 3 and t not in USE_CASE_LIGHT_STOPWORDS and t not in product_tokens
    ]
    for i in range(len(filtered_for_phrase) - 1):
        phrase_terms.add(f"{filtered_for_phrase[i]} {filtered_for_phrase[i + 1]}")

    terms.update(phrase_terms)
    return terms


def has_use_case_alignment(keyword: str, use_case_terms: set) -> bool:
    if not use_case_terms:
        return False
    kw_norm = normalize_keyword(keyword)
    kw_tokens = set(kw_norm.split())

    # Phrase-aware match first (for terms like "joint pain", "pain relief", "blood sugar").
    for term in use_case_terms:
        if " " in term and term in kw_norm:
            return True

    # Token fallback.
    return any(t in kw_tokens for t in use_case_terms if " " not in t)


def is_pack_or_variant_query(keyword: str) -> bool:
    return bool(
        re.search(
            r"\b(pack of \d+|\d+\s*pack|set of \d+|combo|combo pack|\d+\s*(ml|g|gm|kg|l|litre|liter))\b",
            keyword,
        )
    )


def classify_intent_and_funnel(keyword: str, context: ProductContext) -> Tuple[str, str]:
    kw = normalize_keyword(keyword)
    has_buy = contains_any(kw, {"buy", "order", "shop", "online", "purchase"}) or "where to buy" in kw
    has_eval = contains_any(kw, {"best", "top", "review", "reviews", "compare", "comparison", "vs", "which", "price", "cost", "mrp", "rate"})
    has_info = contains_any(kw, INFORMATIONAL_TERMS) or kw.startswith("what ") or kw.startswith("how ")
    has_brand = has_probable_brand_context(kw, context)

    if has_buy:
        return "Transactional", "BOF"
    if has_eval and contains_any(kw, {"review", "reviews", "compare", "comparison", "vs"}):
        return "Commercial", "TOF"
    if has_eval:
        return "Commercial", "MOF"
    if has_info:
        return "Informational", "TOF"
    if has_brand:
        return "Navigational", "MOF"
    return "Commercial", "MOF"


def build_positive_note(keyword: str, intent: str, funnel_stage: str, ad_group: str, context: ProductContext) -> str:
    if is_competitor_product_keyword(keyword, context):
        return f"Competitor + product query kept positive; routed to {ad_group} ({intent}, {funnel_stage})."
    if ad_group.startswith("brand_search_"):
        return f"Brand + product intent; routed to {ad_group} ({intent}, {funnel_stage})."
    if ad_group.startswith("condition_search_"):
        return f"Condition-solution intent aligned to product use case; routed to {ad_group} ({intent}, {funnel_stage})."
    if ad_group.startswith("evaluation_search_"):
        return f"Evaluation/comparison style query; routed to {ad_group} ({intent}, {funnel_stage})."
    if ad_group.startswith("product_search_"):
        return f"Direct product-intent query; routed to {ad_group} ({intent}, {funnel_stage})."
    return f"Classified as {intent} at {funnel_stage}; routed to {ad_group}."


def wrong_format(keyword: str, allowed_format: str) -> bool:
    allowed_canonical = canonicalize_format_token(allowed_format)
    for fmt in FORMAT_TERMS:
        if fmt in keyword and canonicalize_format_token(fmt) != allowed_canonical:
            return True
    return False


def is_cross_product(keyword: str, context: ProductContext) -> bool:
    own = normalize_keyword(context.product_name)
    for pname in context.all_product_names:
        if pname and pname in keyword and pname != own:
            return True
    return False


def step_2_filter_keywords(df: pd.DataFrame, context: ProductContext) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]], List[str]]:
    active: List[Tuple[str, str]] = []
    negative: List[Dict[str, str]] = []
    removed: List[str] = []
    use_case_terms = extract_use_case_terms(context)

    for _, row in df.iterrows():
        kw = row["keyword"]
        fp = str(row.get("Fingerprint", "") or "")
        
        # Step A: Marketplace filtering
        if contains_any(kw, MARKETPLACE_TERMS):
            removed.append(kw)
            intent, funnel = classify_intent_and_funnel(kw, context)
            negative.append({"Keyword": kw, "Fingerprint": fp, "Keyword Polarity": "Negative", "Intent Type": intent, "Funnel Stage": funnel, "Reason for Exclusion": "Marketplace term"})
            continue
            
        # Step B: Non-purchasing term filtering
        if contains_any(kw, NON_PURCHASE_TERMS):
            removed.append(kw)
            intent, funnel = classify_intent_and_funnel(kw, context)
            negative.append({"Keyword": kw, "Fingerprint": fp, "Keyword Polarity": "Negative", "Intent Type": intent, "Funnel Stage": funnel, "Reason for Exclusion": "Non-purchase behavioral query"})
            continue
            
        # Step C: Wrong format filtering
        if wrong_format(kw, context.allowed_format):
            removed.append(kw)
            intent, funnel = classify_intent_and_funnel(kw, context)
            negative.append({"Keyword": kw, "Fingerprint": fp, "Keyword Polarity": "Negative", "Intent Type": intent, "Funnel Stage": funnel, "Reason for Exclusion": "Wrong product format"})
            continue
            
        # Step D: Cross-Product leakage filtering
        if is_cross_product(kw, context):
            removed.append(kw)
            intent, funnel = classify_intent_and_funnel(kw, context)
            negative.append({"Keyword": kw, "Fingerprint": fp, "Keyword Polarity": "Negative", "Intent Type": intent, "Funnel Stage": funnel, "Reason for Exclusion": "Cross-product leakage"})
            continue
            
        # Step E: Ambiguous / off-use-case filtering
        has_product = has_product_reference(kw, context)
        has_commercial = contains_any(kw, TRANSACTIONAL_MODIFIERS | EVALUATION_MODIFIERS)
        has_condition_signal = "for " in kw and ("ayurvedic" in kw or "medicine" in kw or "remedy" in kw)
        aligned_use_case = has_use_case_alignment(kw, use_case_terms)

        is_ambiguous = False
        if not has_product and not has_competitor_term(kw, context):
            if not (aligned_use_case and (has_commercial or has_condition_signal)) and not (aligned_use_case and len(kw.split()) >= 3):
                is_ambiguous = True

        if is_ambiguous:
            removed.append(kw)
            intent, funnel = classify_intent_and_funnel(kw, context)
            negative.append({"Keyword": kw, "Fingerprint": fp, "Keyword Polarity": "Negative", "Intent Type": intent, "Funnel Stage": funnel, "Reason for Exclusion": "Ambiguous/off-primary-use-case intent"})
            continue
            
        # If it passes all filters, it is active
        active.append((kw, fp))

    return active, negative, removed


def classify_ad_group(kw: str, context: ProductContext) -> Tuple[str, str]:
    has_brand = has_probable_brand_context(kw, context)
    has_product = has_product_reference(kw, context)
    has_competitor = any(c in kw for c in context.competitors)
    has_tx = contains_any(kw, TRANSACTIONAL_MODIFIERS)

    if has_competitor and has_tx:
        return "Transactional", f"competitor_search_{context.product_slug}"
    if has_brand and has_product:
        return "Navigational", f"brand_search_{context.product_slug}"
    if contains_any(kw, EVALUATION_MODIFIERS):
        return "Evaluation", f"evaluation_search_{context.product_slug}"
    if (
        ("for " in kw and ("ayurvedic" in kw or "medicine" in kw or "remedy" in kw or context.allowed_format in kw))
        or ("best ayurvedic" in kw)
    ):
        return "Transactional", f"condition_search_{context.product_slug}"
    return "Transactional", f"product_search_{context.product_slug}"


def step_3_classification(active_keywords: List[Tuple[str, str]], context: ProductContext, llm_client: OpenAI, llm_model: str) -> Tuple[List[KeywordRecord], List[Dict[str, str]]]:
    if not active_keywords:
        return [], []
    kw_list = [k for k, _ in active_keywords]
    fp_by_kw = {k: fp for k, fp in active_keywords}
    batch_size = 5
    total_batches = (len(kw_list) + batch_size - 1) // batch_size
    allowed_intents = {"Informational", "Commercial", "Transactional", "Navigational"}
    allowed_groups = {
        "product_search",
        "brand_search",
        "condition_search",
        "evaluation_search",
        "competitor_search",
    }

    def classify_batch(keywords: List[str]) -> Dict[str, dict]:
        system_prompt = (
            "You classify Google Ads keywords. Return strict JSON only with key 'results'. "
            "Each item must be: keyword, intent_type (Informational|Commercial|Transactional|Navigational), "
            "funnel_stage (BOF|MOF|TOF), "
            "ad_group_base (product_search|brand_search|condition_search|evaluation_search|competitor_search). "
            "INTENT DEFINITIONS:\n"
            "- Informational: Educational or research intent (symptoms, causes, benefits, uses, ingredients, side effects, meaning, definitions, how-to). Includes product-specific info queries like '[product] uses' and '[product] ingredients' and seek a specific medicine to buy (e.g., 'how to lower blood sugar', 'what is diabetes', 'symptoms of diabetes', 'how to use [product]', 'what is [product]', 'what are the ingredients of [product]', 'what are the uses of [product]', 'how to use [product]).\n"
            "- Commercial: Comparing options, looking for reviews, checking prices, OR searching for a generic solution/medicine to treat a condition (e.g., 'best ayurvedic tablet for diabetes', 'ayurvedic medicine to lower blood sugar').\n"
            "- Transactional: Clear intent to buy right now (e.g., 'buy [product]', 'ayurvedic diabetes medicine online').\n"
            "- Navigational: Searching specifically for the brand/product name directly without modifiers (e.g., '[brand] [product]').\n"
            "PRIORITY RULE: If a query contains clear purchase signals (buy/order/price/cost/shop/online/purchase), do NOT classify it as Informational.\n"
            "COMPETITOR RULES AND EXAMPLES:\n"
            "1. Only competitor brand (e.g., 'dabur') -> MUST be Informational.\n"
            "2. Competitor brand + product name (e.g., 'dabur vs [product]', 'dabur [product]') -> MUST be Commercial (with ad_group_base competitor_search).\n"
            "3. Competitor brand + product name + ingredients/informational terms (e.g., 'dabur [product] ingredients') -> MUST be Informational.\n"
            "4. Competitor brand + product use case intent (e.g., 'dabur diabetes medicine', 'patanjali sugar medicine', 'dabur ayurvedic sugar medicine' ) -> MUST be Commercial (with ad_group_base competitor_search).\n"
            "If a keyword contains a competitor term and product reference, ad_group_base must be competitor_search. "
            "Return exactly one item for every input keyword."
        )
        user_prompt = (
            f"Product: {context.product_name}\n"
            f"Product website: {context.product_url}\n"
            f"Competitors: {', '.join(sorted(competitor_terms(context))) if competitor_terms(context) else 'None'}\n"
            f"Allowed format: {context.allowed_format}\n"
            "Keywords:\n- " + "\n- ".join(keywords)
        )
        data = chat_json(llm_client, llm_model, system_prompt, user_prompt)
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ValueError("LLM classification response missing required 'results' list")
        return {normalize_keyword(str(x.get("keyword", ""))): x for x in data["results"] if isinstance(x, dict)}

    by_kw: Dict[str, dict] = {}
    for i in range(0, len(kw_list), batch_size):
        batch = kw_list[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"[Progress] Step 3/9 Classification: batch {batch_num}/{total_batches} ({len(batch)} keywords)")
        result = classify_batch(batch)
        missing = [kw for kw in batch if kw not in result]
        if missing:
            print(f"[Progress] Step 3/9 Classification: retrying {len(missing)} missing keywords in batch {batch_num}")
            retry_result = classify_batch(missing)
            result.update(retry_result)
            missing = [kw for kw in batch if kw not in result]
            if missing:
                raise ValueError(f"LLM classification missing {len(missing)} keywords after retry in batch")
        for kw in batch:
            by_kw[kw] = result[kw]

    rows: List[KeywordRecord] = []
    informational_negatives: List[Dict[str, str]] = []
    use_case_terms = extract_use_case_terms(context)
    for kw in kw_list:
        picked = by_kw[kw]
        intent = str(picked.get("intent_type", "")).strip()
        funnel_stage = str(picked.get("funnel_stage", "")).strip()
        ad_group_base = str(picked.get("ad_group_base", "")).strip()
        if has_competitor_term(kw, context):
            ad_group_base = "competitor_search"
        if intent not in allowed_intents:
            raise ValueError(f"LLM classification returned invalid intent '{intent}' for keyword '{kw}'")
        if funnel_stage not in {"BOF", "MOF", "TOF"}:
            raise ValueError(f"LLM classification returned invalid funnel_stage '{funnel_stage}' for keyword '{kw}'")
        if ad_group_base not in allowed_groups:
            raise ValueError(f"LLM classification returned invalid ad_group_base '{ad_group_base}' for keyword '{kw}'")
            
        has_buy_signal = contains_any(kw, TRANSACTIONAL_MODIFIERS | {"cost", "mrp", "rate"})
        has_info_signal = contains_any(kw, INFORMATIONAL_TERMS) or kw.startswith("what ") or kw.startswith("how ")
        has_hard_info_negative = contains_any(kw, HARD_INFORMATIONAL_NEGATIVE_TERMS) or is_pack_or_variant_query(kw)
        aligned_use_case = has_use_case_alignment(kw, use_case_terms)
        has_category_signal = contains_any(kw, {"ayurvedic", "oil", context.allowed_format})
        if has_info_signal:
            intent = "Informational"
            funnel_stage = "TOF"

        # Use-case rescue: keep validated product use-case demand positive unless it's explicitly review/pack informational.
        if (
            intent == "Informational"
            and not has_info_signal
            and not has_hard_info_negative
            and aligned_use_case
            and has_category_signal
            and not has_competitor_term(kw, context)
        ):
            intent = "Commercial"
            funnel_stage = "MOF"
            ad_group_base = "condition_search"

        if has_hard_info_negative:
            intent = "Informational"
            funnel_stage = "TOF"

        if intent == "Informational":
            informational_negatives.append({
                "Keyword": kw,
                "Fingerprint": fp_by_kw.get(kw, ""),
                "Keyword Polarity": "Negative",
                "Intent Type": intent,
                "Funnel Stage": funnel_stage,
                "Reason for Exclusion": "Informational intent from LLM/rule",
            })
            continue

        ad_group = f"{ad_group_base}_{context.product_slug}"
        rows.append(
            KeywordRecord(
                keyword=kw,
                fingerprint=fp_by_kw.get(kw, ""),
                keyword_polarity="Positive",
                match_type="",
                intent_type=intent,
                funnel_stage=funnel_stage,
                ad_group=ad_group,
                notes=build_positive_note(kw, intent, funnel_stage, ad_group, context),
            )
        )
    return rows, informational_negatives


def step_4_match_types(records: List[KeywordRecord], context: ProductContext) -> List[KeywordRecord]:
    broad_candidates: List[int] = []
    for i, rec in enumerate(records):
        kw = rec.keyword
        has_competitor = has_competitor_term(kw, context)
        has_brand_product = "kerala ayurveda" in kw and normalize_keyword(context.product_name) in kw
        has_product_tx = normalize_keyword(context.product_name) in kw and contains_any(kw, TRANSACTIONAL_MODIFIERS)

        if has_competitor and contains_any(kw, TRANSACTIONAL_MODIFIERS):
            rec.match_type = "Exact Match"
        elif has_brand_product or has_product_tx:
            rec.match_type = "Exact Match"
        elif rec.ad_group.startswith("condition_") or rec.ad_group.startswith("evaluation_"):
            rec.match_type = "Phrase Match"
        elif rec.ad_group.startswith("product_") and not contains_any(kw, TRANSACTIONAL_MODIFIERS):
            broad_candidates.append(i)
            rec.match_type = "Broad Match"
        else:
            rec.match_type = "Phrase Match"

    broad_cap = int(len(records) * 0.10)
    if broad_cap < len([r for r in records if r.match_type == "Broad Match"]):
        keep = set(broad_candidates[:broad_cap])
        for i in broad_candidates:
            if i not in keep:
                records[i].match_type = "Phrase Match"
    return records


def step_6_filter_confirmation(records: List[KeywordRecord]) -> None:
    allowed = {"Commercial", "Transactional", "Navigational"}
    for rec in records:
        if rec.intent_type not in allowed:
            raise ValueError(f"Invalid intent type in active list: {rec.keyword} -> {rec.intent_type}")


def step_7_brand_rules(records: List[KeywordRecord]) -> Tuple[List[KeywordRecord], List[Dict[str, str]]]:
    output: List[KeywordRecord] = []
    brand_negatives: List[Dict[str, str]] = []
    for r in records:
        if normalize_keyword(r.keyword) == "kerala ayurveda":
            brand_negatives.append({
                "Keyword": r.keyword,
                "Fingerprint": r.fingerprint,
                "Keyword Polarity": "Negative",
                "Intent Type": r.intent_type,
                "Funnel Stage": r.funnel_stage,
                "Reason for Exclusion": "Brand rule: standalone brand term with no product reference",
            })
        else:
            output.append(r)
    return output, brand_negatives


def step_8_competitor_rules(records: List[KeywordRecord], context: ProductContext) -> Tuple[List[KeywordRecord], List[Dict[str, str]]]:
    out: List[KeywordRecord] = []
    competitor_negatives: List[Dict[str, str]] = []
    for r in records:
        has_comp = has_competitor_term(r.keyword, context)
        if has_comp:
            if r.intent_type in {"Commercial", "Transactional", "Navigational"}:
                r.ad_group = f"competitor_search_{context.product_slug}"
                out.append(r)
                continue
            competitor_negatives.append({
                "Keyword": r.keyword,
                "Fingerprint": r.fingerprint,
                "Keyword Polarity": "Negative",
                "Intent Type": r.intent_type,
                "Funnel Stage": r.funnel_stage,
                "Reason for Exclusion": "Competitor rule: informational competitor query",
            })
            continue
        out.append(r)
    return out, competitor_negatives


def step_9_build_campaign_structure(context: ProductContext) -> pd.DataFrame:
    campaign = f"google_search_products_{context.product_slug}"
    rows = [
        (campaign, f"product_search_{context.product_slug}", "Transactional", "Pure product name and transactional product variants"),
        (campaign, f"brand_search_{context.product_slug}", "Navigational", "Kerala Ayurveda + product combinations"),
        (campaign, f"condition_search_{context.product_slug}", "Transactional", "Condition + ayurvedic solution intent keywords"),
        (campaign, f"evaluation_search_{context.product_slug}", "Evaluation", "Best, review, comparison intent keywords"),
        (campaign, f"competitor_search_{context.product_slug}", "Transactional", "Competitor + product + buy/price/online/order keywords"),
    ]
    return pd.DataFrame(rows, columns=["Campaign Name", "Ad Group Name", "Intent Type", "Description"])


def step_10_quality_checks(records: List[KeywordRecord]) -> None:
    pairs = [(r.keyword, r.ad_group) for r in records]
    if len({p[0] for p in pairs}) != len(pairs):
        raise ValueError("QA failed: duplicate keyword found across ad groups")

    broad_count = sum(1 for r in records if r.match_type == "Broad Match")
    if records and broad_count / len(records) > 0.10:
        raise ValueError("QA failed: Broad Match exceeds 10%")

    for r in records:
        if contains_any(r.keyword, INFORMATIONAL_TERMS):
            raise ValueError(f"QA failed: informational keyword in active list: {r.keyword}")


def enforce_broad_match_cap(records: List[KeywordRecord], max_share: float = 0.10) -> List[KeywordRecord]:
    if not records:
        return records
    broad_indices = [i for i, r in enumerate(records) if r.match_type == "Broad Match"]
    allowed = int(len(records) * max_share)
    if len(broad_indices) <= allowed:
        return records
    demote_count = len(broad_indices) - allowed
    # Demote later broad entries first to preserve earlier priority ordering.
    for i in broad_indices[-demote_count:]:
        records[i].match_type = "Phrase Match"
    return records


def export_output(
    output_path: Path,
    campaign_structure: pd.DataFrame,
    records: List[KeywordRecord],
    negative: List[Dict[str, str]],
) -> None:
    def ensure_note(row: pd.Series) -> str:
        note = str(row.get("Notes", "") or "").strip()
        if note:
            return note
        polarity = str(row.get("Keyword Polarity", "") or "").strip()
        intent = str(row.get("Intent Type", "") or "").strip()
        funnel = str(row.get("Funnel Stage", "") or "").strip()
        ad_group = str(row.get("Ad Group", "") or "").strip()
        if polarity == "Negative":
            return f"Marked negative for campaign hygiene ({intent}, {funnel})."
        return f"Classified as positive in {ad_group} ({intent}, {funnel})."

    positive_df = pd.DataFrame(
        [
            {
                "Keyword": r.keyword,
                "Fingerprint": r.fingerprint,
                "Keyword Polarity": r.keyword_polarity,
                "Match Type": r.match_type,
                "Intent Type": r.intent_type,
                "Funnel Stage": r.funnel_stage,
                "Ad Group": r.ad_group,
                "Notes": r.notes,
                "English Equivalent": r.english_equivalent,
                "Original Language": r.original_language,
            }
            for r in records
        ]
    )

    negative_df = pd.DataFrame(negative)
    if negative_df.empty:
        negative_df = pd.DataFrame(
            columns=["Keyword", "Fingerprint", "Keyword Polarity", "Intent Type", "Funnel Stage", "Reason for Exclusion"]
        )
    negative_df["Match Type"] = "Negative Match"
    negative_df["Ad Group"] = "negative_keywords"
    negative_df["Notes"] = negative_df["Reason for Exclusion"]
    negative_df["English Equivalent"] = ""
    negative_df["Original Language"] = ""
    negative_df = negative_df.drop(columns=["Reason for Exclusion"])

    keywords_df = pd.concat([positive_df, negative_df], ignore_index=True, sort=False)
    keywords_df = keywords_df[
        ["Keyword", "Fingerprint", "Keyword Polarity", "Match Type", "Intent Type", "Funnel Stage", "Ad Group", "Notes", "English Equivalent", "Original Language"]
    ]
    keywords_df["Notes"] = keywords_df.apply(ensure_note, axis=1)

    fingerprint_export_df = keywords_df.copy()
    if not fingerprint_export_df.empty:
        unique_fps = [fp for fp in fingerprint_export_df["Fingerprint"].fillna("").astype(str).unique().tolist() if fp]
        fp_ids = {fp: f"fp_{i + 1:04d}" for i, fp in enumerate(unique_fps)}
        fingerprint_export_df["Fingerprint Group ID"] = fingerprint_export_df["Fingerprint"].fillna("").astype(str).map(fp_ids)
    else:
        fingerprint_export_df["Fingerprint Group ID"] = ""

    seen_fp_first_keyword: Dict[str, str] = {}
    for idx, row in fingerprint_export_df.iterrows():
        fp = str(row.get("Fingerprint", "") or "").strip()
        kw = str(row.get("Keyword", "") or "").strip()
        if not fp:
            continue
        if fp not in seen_fp_first_keyword:
            seen_fp_first_keyword[fp] = kw
            continue

        first_kw = seen_fp_first_keyword[fp]
        fingerprint_export_df.at[idx, "Keyword Polarity"] = "Negative"
        fingerprint_export_df.at[idx, "Match Type"] = "Negative Match"
        fingerprint_export_df.at[idx, "Ad Group"] = "negative_keywords"
        existing_note = str(fingerprint_export_df.at[idx, "Notes"] or "").strip()
        dedup_note = f"Duplicate fingerprint group; kept '{first_kw}' as primary and marked this as negative."
        fingerprint_export_df.at[idx, "Notes"] = dedup_note if not existing_note else f"{existing_note} | {dedup_note}"

    # Ensure first-of-group and no-fingerprint rows also always have notes.
    fingerprint_export_df["Notes"] = fingerprint_export_df.apply(ensure_note, axis=1)

    valid_fps = fingerprint_export_df[fingerprint_export_df["Fingerprint Group ID"] != ""]
    if not valid_fps.empty:
        summary_df = valid_fps.groupby(["Fingerprint Group ID", "Fingerprint"]).agg(
            Keyword_Count=("Keyword", "count"),
            All_Keywords=("Keyword", lambda x: ", ".join(list(x)))
        ).reset_index()
        summary_df = summary_df.sort_values("Fingerprint Group ID", ascending=True)
    else:
        summary_df = pd.DataFrame(columns=["Fingerprint Group ID", "Fingerprint", "Keyword_Count", "All_Keywords"])

    fingerprint_export_df = fingerprint_export_df[
        [
            "Keyword",
            "Fingerprint Group ID",
            "Fingerprint",
            "Keyword Polarity",
            "Match Type",
            "Intent Type",
            "Funnel Stage",
            "Ad Group",
            "Notes",
            "English Equivalent",
            "Original Language",
        ]
    ]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        campaign_structure.to_excel(writer, sheet_name="CAMPAIGN_STRUCTURE", index=False)
        keywords_df.to_excel(writer, sheet_name="KEYWORDS", index=False)
        fingerprint_export_df.to_excel(writer, sheet_name="FINGERPRINT_DEDUP", index=False)
        summary_df.to_excel(writer, sheet_name="FINGERPRINT_SUMMARY", index=False)


def process_product(
    product_name: str,
    input_csv: Path,
    positioning_xlsx: Path,
    output_dir: Path,
    llm_config: LLMConfig,
) -> Path:
    print("[Progress] Initializing LLM client and loading product context")
    llm_client = build_llm_client(llm_config)
    context = load_positioning_context(positioning_xlsx, product_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_xlsx = output_dir / f"{small_product_name(context.product_name)}.xlsx"
    print("[Progress] Running startup checks")
    step_0_startup_checks(context)

    print("[Progress] Loading input keywords from CSV")
    raw_df = load_product_keywords(input_csv, product_name)
    total_input_count = len(raw_df)

    print("[Progress] Step 1/9 Keyword cleaning")
    cleaned_df, cleaning_negatives = step_1_keyword_cleaning(raw_df)

    print("[Progress] Step 2/9 Filtering keywords")
    active, negative, _removed = step_2_filter_keywords(cleaned_df, context)
    negative.extend(cleaning_negatives)

    print("[Progress] Step 3/9 LLM classification")
    records, pure_info_negatives = step_3_classification(active, context, llm_client, llm_config.model)
    negative.extend(pure_info_negatives)

    print("[Progress] Step 3b/9 Language detection & English translation")
    records = translate_keywords_batch(records, llm_client, llm_config.model)
    print("[Progress] Step 3c/9 Canonicalizing English equivalents for semantic grouping")
    records = canonicalize_english_equivalents_batch(records, llm_client, llm_config.model)

    print("[Progress] Step 4/9 Fingerprint semantic deduplication")
    records = step_1b_fingerprint_semantic_dedup(records, context, llm_client, llm_config.model)

    print("[Progress] Step 5/9 Match type assignment")
    records = step_4_match_types(records, context)

    print("[Progress] Step 6/9 Filter confirmation")
    step_6_filter_confirmation(records)

    print("[Progress] Step 7/9 Brand rules")
    records, brand_negatives = step_7_brand_rules(records)
    negative.extend(brand_negatives)

    print("[Progress] Step 8/9 Competitor rules")
    records, competitor_negatives = step_8_competitor_rules(records, context)
    negative.extend(competitor_negatives)

    records = enforce_broad_match_cap(records, max_share=0.10)

    print("[Progress] Step 9/9 Campaign structure build")
    campaign_structure = step_9_build_campaign_structure(context)

    print("[Progress] Quality checks")
    step_10_quality_checks(records)

    # Reconciliation: ensure zero keyword loss
    total_output_count = len(records) + len(negative)
    if total_input_count != total_output_count:
        raise ValueError(
            f"Keyword count reconciliation FAILED: {total_input_count} keywords in, "
            f"{total_output_count} keywords out. Difference = {total_input_count - total_output_count}. "
            f"All keywords must be preserved — none should be silently deleted."
        )
    print(f"[Progress] Reconciliation passed: {total_input_count} keywords in = {total_output_count} keywords out")

    print("[Progress] Exporting output workbook")
    export_output(output_xlsx, campaign_structure, records, negative)
    return output_xlsx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kerala Ayurveda keyword cleaner and classifier")
    parser.add_argument("--product", required=True, help="Product name exactly as in positioning document")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    llm_config = LLMConfig(
        base_url=OPENROUTER_BASE_URL,
        model=OPENROUTER_MODEL,
        api_key_env=OPENROUTER_API_KEY_ENV,
    )
    output_xlsx = process_product(
        product_name=args.product,
        input_csv=INPUT_KEYWORDS_CSV,
        positioning_xlsx=POSITIONING_XLSX,
        output_dir=OUTPUT_DIR,
        llm_config=llm_config,
    )
    print(f"Processed '{args.product}' -> {output_xlsx}")


if __name__ == "__main__":
    main()
