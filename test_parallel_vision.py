#!/usr/bin/env python3
"""
Test parallel vision API with a PDF that likely has figures.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from paper_sailor.tools.multimodal_parser import extract_figures_and_tables


def main():
    print("=" * 80)
    print("Parallel Vision API Test")
    print("=" * 80)
    
    # Try multiple PDFs to find ones with figures
    test_pdfs = [
        "data/pdfs/arxiv:2511.04093v1.pdf",
        "data/pdfs/arxiv:2509.22028v1.pdf",
        "data/pdfs/arxiv:2410.07981v2.pdf",
        "data/pdfs/openalex:W2939547055.pdf",
        "data/pdfs/arxiv:2510.11709v1.pdf",
    ]
    
    for pdf_name in test_pdfs:
        pdf_path = Path(pdf_name)
        if not pdf_path.exists():
            continue
        
        paper_id = pdf_path.stem
        print(f"\n{'='*80}")
        print(f"Testing: {pdf_path.name}")
        print(f"{'='*80}")
        
        try:
            # Extract with parallel processing
            results = extract_figures_and_tables(
                str(pdf_path),
                paper_id,
                verbose=True,
                max_pages=5,  # First 5 pages
                extract_tables=False,  # Only figures
                max_workers=6  # 6 parallel workers
            )
            
            if results:
                print(f"\n✅ SUCCESS: Found {len(results)} figures")
                print(f"\nSample descriptions:")
                for i, item in enumerate(results[:3], 1):
                    desc = item.get("visual_description", "N/A")
                    print(f"{i}. Page {item.get('page_from')}: {desc[:100]}...")
                
                # We found a PDF with figures, stop here
                return True
            else:
                print(f"⏭️  No figures found in first 5 pages, trying next PDF...")
        
        except Exception as exc:
            print(f"❌ Error: {exc}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("⚠️  None of the test PDFs had figures in the first 5 pages")
    print("=" * 80)
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

