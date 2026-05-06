import argparse
import pandas as pd
from pathlib import Path

def prepare_input(product_name: str):
    raw_input_dir = Path("raw_input")
    raw_file = raw_input_dir / f"{product_name}.xlsx"
    input_dir = Path("input")
    output_file = input_dir / "input_keywords.csv"

    if not raw_file.exists():
        raise FileNotFoundError(f"Raw keyword file not found: {raw_file}")

    print(f"Reading raw keywords from {raw_file}...")
    # Read the raw excel file
    df = pd.read_excel(raw_file)
    
    # Ensure there is a 'keyword' column (case-insensitive)
    col_map = {str(c).strip().lower(): c for c in df.columns}
    keyword_col = col_map.get("keyword")
    
    if not keyword_col:
        raise ValueError(f"Raw file {raw_file} is missing the required 'keyword' header.")
        
    keywords = df[keyword_col].dropna().astype(str).tolist()
    
    print(f"Extracted {len(keywords)} keywords.")
    
    # Create input directory if it doesn't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Delete the old input_keywords.csv to avoid data leakage
    if output_file.exists():
        print(f"Deleting existing file: {output_file}")
        output_file.unlink()
        
    # Create the new DataFrame
    new_df = pd.DataFrame({
        "product_name": [product_name] * len(keywords),
        "keyword": keywords
    })
    
    print(f"Saving formatted keywords to {output_file}...")
    new_df.to_csv(output_file, index=False)
    print("Done! You can now run main.py.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform raw product keywords for main.py")
    parser.add_argument("--product", required=True, help="Product name (also used to find the .xlsx file)")
    args = parser.parse_args()
    
    prepare_input(args.product)
