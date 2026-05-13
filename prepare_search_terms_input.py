import argparse
from pathlib import Path

import pandas as pd


RAW_INPUT_DIR = Path("raw_input")
OUTPUT_CSV = Path("input/input_search_terms.csv")
SEARCH_TERM_HEADERS = ["search term", "search terms", "query", "term"]


def normalize_header(value: object) -> str:
    return str(value).strip().lower()


def find_search_term_column(df: pd.DataFrame) -> str:
    col_map = {normalize_header(c): c for c in df.columns}
    for h in SEARCH_TERM_HEADERS:
        if h in col_map:
            return col_map[h]
    for c in df.columns:
        series = df[c].dropna().astype(str).str.strip()
        if not series.empty:
            return c
    raise ValueError("No usable search-term column found.")


def prepare_search_term_input(product_name: str) -> Path:
    raw_file = RAW_INPUT_DIR / f"{product_name} Search Term.xlsx"
    if not raw_file.exists():
        raise FileNotFoundError(f"Search-term file not found: {raw_file}")

    df = pd.read_excel(raw_file)
    col = find_search_term_column(df)
    values = [str(v).strip() for v in df[col].dropna().tolist()]
    values = [v for v in values if v]
    if not values:
        raise ValueError(f"No valid search terms found in {raw_file}")

    out_df = pd.DataFrame(
        {
            "product_name": [product_name] * len(values),
            "keyword": values,
        }
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()
    out_df.to_csv(OUTPUT_CSV, index=False)
    return OUTPUT_CSV


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare input/input_search_terms.csv from raw search-term file.")
    parser.add_argument("--product", required=True, help="Product name (base filename in raw_input).")
    args = parser.parse_args()
    out = prepare_search_term_input(args.product)
    print(f"Prepared: {out}")


if __name__ == "__main__":
    main()
