import argparse
import sys
from pathlib import Path

from groq_extract import extract_understanding, get_client
from vector_db import VectorDB

def main():
    ap = argparse.ArgumentParser(description="Extract image metadata using Groq and store in Vector DB.")
    ap.add_argument("images", nargs="+", help="Image file(s) to process.")
    ap.add_argument("--model", default="qwen/qwen3.6-27b", help="Groq vision model ID.")
    args = ap.parse_args()

    client = get_client()
    db = VectorDB()

    for p in args.images:
        path = Path(p)
        if not path.exists():
            print(f"!! file not found: {p}", file=sys.stderr)
            continue
            
        print(f"Processing {p}...")
        try:
            # Step 1: Extract validated ImageUnderstanding schema
            understanding = extract_understanding(str(path), model=args.model, client=client)
            
            # Step 2: Store in Vector DB
            db.add_image_understanding(image_id=str(path), understanding=understanding)
            print(f"  -> Successfully extracted and stored: {understanding.subject}")
        except Exception as e:
            print(f"  -> Failed to process {p}: {e}")

if __name__ == "__main__":
    main()
