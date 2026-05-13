import argparse

from build_search_term_manager_outputs import build_outputs
from main_search_terms import (
    LLMConfig,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OUTPUT_DIR as CLASSIFIED_DIR,
    POSITIONING_XLSX,
    process_product,
)
from prepare_search_terms_input import prepare_search_term_input


def run_pipeline(product_name: str) -> None:
    prepared_csv = prepare_search_term_input(product_name)
    print(f"[1/3] Prepared search-term CSV: {prepared_csv}")

    llm_config = LLMConfig(
        base_url=OPENROUTER_BASE_URL,
        model=OPENROUTER_MODEL,
        api_key_env=OPENROUTER_API_KEY_ENV,
    )
    classified_output = process_product(
        product_name=product_name,
        input_csv=prepared_csv,
        positioning_xlsx=POSITIONING_XLSX,
        output_dir=CLASSIFIED_DIR,
        llm_config=llm_config,
    )
    print(f"[2/3] Classified search terms output: {classified_output}")

    pos, neg, neg_kw = build_outputs(product_name)
    print(f"[3/3] Positive search-term file: {pos}")
    print(f"[3/3] Negative search-term file: {neg}")
    print(f"[3/3] Negative keyword file: {neg_kw}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end search-term pipeline using main_search_terms classifier plus manager output builder."
    )
    parser.add_argument("--product", required=True, help="Product name.")
    args = parser.parse_args()
    run_pipeline(args.product)


if __name__ == "__main__":
    main()
