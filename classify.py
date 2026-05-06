import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd


DEFAULT_KEYWORDS_DIR = Path("keywords")
DEFAULT_KW_DUMP_DIR = Path("kw dump")
DEFAULT_OUTPUT = Path("classified_keywords.xlsx")
KEYWORD_HEADER = "keyword"
SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}


def normalize_keyword(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def display_keyword(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def is_ignored_keywords_path(path: Path, keywords_dir: Path) -> bool:
    try:
        relative_parts = path.relative_to(keywords_dir).parts
    except ValueError:
        relative_parts = path.parts
    return any(part.strip().lower() == "with ads" for part in relative_parts)


def iter_input_files(root: Path, *, ignore_with_ads: bool = False) -> Iterable[Path]:
    if not root.exists():
        return

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if ignore_with_ads and is_ignored_keywords_path(path, root):
            continue
        yield path


def product_name_from_path(path: Path) -> str:
    name = path.stem
    name = re.sub(r"_keyword_campaign$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_updated$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_RSA_Copy(?:_v\d+)?$", "", name, flags=re.IGNORECASE)
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def extract_from_frame(frame: pd.DataFrame) -> List[str]:
    keywords: List[str] = []

    for row_index, row in frame.iterrows():
        keyword_columns = [
            column_index
            for column_index, value in row.items()
            if normalize_keyword(value) == KEYWORD_HEADER
        ]
        if not keyword_columns:
            continue

        for column_index in keyword_columns:
            values = frame.loc[row_index + 1 :, column_index]
            for value in values:
                keyword = display_keyword(value)
                if keyword:
                    keywords.append(keyword)

    return keywords


def extract_keywords(path: Path) -> List[str]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, header=None)
        return extract_from_frame(frame)

    keywords: List[str] = []
    sheets: Dict[str, pd.DataFrame] = pd.read_excel(path, sheet_name=None, header=None)
    for frame in sheets.values():
        keywords.extend(extract_from_frame(frame))
    return keywords


def gather_records(root: Path, *, ignore_with_ads: bool = False) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    seen: Set[tuple] = set()

    for path in iter_input_files(root, ignore_with_ads=ignore_with_ads):
        product_name = product_name_from_path(path)
        for keyword in extract_keywords(path):
            key = normalize_keyword(keyword)
            if not key:
                continue
            row_key = (product_name.lower(), key)
            if row_key in seen:
                continue
            seen.add(row_key)
            records.append(
                {
                    "product_name": product_name,
                    "keyword": keyword,
                    "normalized_keyword": key,
                }
            )

    return records


def classify_keywords(
    positive_records: List[Dict[str, str]],
    dump_records: List[Dict[str, str]],
) -> pd.DataFrame:
    positive_keys = {record["normalized_keyword"] for record in positive_records}
    all_records = positive_records + dump_records
    seen: Set[tuple] = set()
    rows = []

    for record in all_records:
        key = record["normalized_keyword"]
        row_key = (record["product_name"].lower(), key)
        if row_key in seen:
            continue
        seen.add(row_key)
        rows.append(
            {
                "product_name": record["product_name"],
                "keyword": record["keyword"],
                "classification": "positive" if key in positive_keys else "negative",
            }
        )

    return pd.DataFrame(rows).sort_values(["product_name", "classification", "keyword"])


def write_output(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in {".xlsx", ".xls"}:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="classified_keywords")
    else:
        frame.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify keywords from kw dump as positive when they also appear in the "
            "keywords folder. The keywords/with ads folder is ignored."
        )
    )
    parser.add_argument("--keywords-dir", type=Path, default=DEFAULT_KEYWORDS_DIR)
    parser.add_argument("--kw-dump-dir", type=Path, default=DEFAULT_KW_DUMP_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    positive_records = gather_records(args.keywords_dir, ignore_with_ads=True)
    dump_records = gather_records(args.kw_dump_dir)
    classified = classify_keywords(positive_records, dump_records)
    write_output(classified, args.output)

    total = len(classified)
    positives = int((classified["classification"] == "positive").sum()) if total else 0
    negatives = total - positives
    print(f"Wrote {total} keywords to {args.output}")
    print(f"positive: {positives}")
    print(f"negative: {negatives}")


if __name__ == "__main__":
    main()
