#!/usr/bin/env python3
"""
Quick multimodal test - processes only first 3 pages for speed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from paper_sailor.tools.multimodal_parser import extract_figures_and_tables
from paper_sailor.tools.embeddings import embed_multimodal
from paper_sailor.tools.retrieval import multimodal_retrieve
from paper_sailor.vectorstore import VectorStore
from paper_sailor.memory import MemoryManager


def quick_test():
    print("=" * 80)
    print("Quick Multimodal Test (First 3 Pages Only)")
    print("=" * 80)
    
    # Test PDF
    test_pdf = Path("data/pdfs/arxiv:2510.11709v1.pdf")
    paper_id = "arxiv:2510.11709v1"
    session_id = "quick_test_multimodal"
    
    if not test_pdf.exists():
        print(f"❌ Test PDF not found: {test_pdf}")
        return False
    
    print(f"\n📄 Test PDF: {test_pdf}")
    print(f"📋 Processing: First 3 pages only (for speed)")
    
    # Step 1: Extract figures and tables
    print("\n" + "=" * 80)
    print("STEP 1: Extracting visuals from PDF (first 3 pages)...")
    print("=" * 80)
    
    try:
        # Only extract figures (not tables), process first 3 pages, use 4 parallel workers
        visual_items = extract_figures_and_tables(
            str(test_pdf), 
            paper_id, 
            verbose=True,
            max_pages=3,
            extract_tables=False,  # Only figures
            max_workers=4  # Parallel processing
        )
        
        print(f"\n✅ Extracted {len(visual_items)} visual items")
        
        figures = [item for item in visual_items if item.get("content_type") == "figure"]
        tables = [item for item in visual_items if item.get("content_type") == "table"]
        
        print(f"   - Figures: {len(figures)}")
        print(f"   - Tables: {len(tables)}")
        
        if visual_items:
            print("\nSample items:")
            for i, item in enumerate(visual_items[:3], 1):
                print(f"\n   {i}. Type: {item.get('content_type')}")
                print(f"      ID: {item.get('id')}")
                desc = item.get('visual_description', item.get('text', 'N/A'))
                print(f"      Description: {desc[:100]}...")
        
        if not visual_items:
            print("\n⚠️  No visuals extracted, using synthetic test data...")
            visual_items = [
                {
                    "id": f"{paper_id}:test:0001",
                    "paper_id": paper_id,
                    "section": "Figure",
                    "page_from": 1,
                    "page_to": 1,
                    "text": "Test architecture diagram",
                    "content_type": "figure",
                    "visual_description": "Model architecture with encoder-decoder"
                }
            ]
    except Exception as exc:
        print(f"❌ Extraction failed: {exc}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Test embeddings
    print("\n" + "=" * 80)
    print("STEP 2: Testing embeddings...")
    print("=" * 80)
    
    try:
        items_to_embed = []
        for item in visual_items[:3]:  # Limit for speed
            content = item.get("visual_description") or item.get("text", "")
            if content:
                items_to_embed.append({
                    "type": item.get("content_type", "text"),
                    "content": content
                })
        
        print(f"📊 Embedding {len(items_to_embed)} items...")
        embeddings = embed_multimodal(items_to_embed)
        print(f"✅ Generated {len(embeddings)} embeddings (dim: {len(embeddings[0]) if embeddings else 0})")
    except Exception as exc:
        print(f"❌ Embedding failed: {exc}")
        return False
    
    # Step 3: Store and retrieve
    print("\n" + "=" * 80)
    print("STEP 3: Testing vector store...")
    print("=" * 80)
    
    try:
        store = VectorStore()
        store.delete_session(session_id)
        
        records = []
        for item, emb in zip(visual_items[:len(embeddings)], embeddings):
            records.append({
                "chunk_id": item["id"],
                "paper_id": item.get("paper_id"),
                "text": item.get("text", ""),
                "embedding": emb,
                "content_type": item.get("content_type", "text"),
                "visual_description": item.get("visual_description"),
                "metadata": {
                    "section": item.get("section"),
                    "page_from": item.get("page_from"),
                    "page_to": item.get("page_to"),
                }
            })
        
        store.upsert_multimodal(session_id, records)
        print(f"✅ Stored {len(records)} records")
    except Exception as exc:
        print(f"❌ Vector store failed: {exc}")
        return False
    
    # Step 4: Test retrieval
    print("\n" + "=" * 80)
    print("STEP 4: Testing multimodal retrieval...")
    print("=" * 80)
    
    try:
        memory_manager = MemoryManager()
        memory_manager.add_session_context(session_id, {
            "topic": "multimodal test",
            "paper_id": paper_id
        })
        
        query = "What figures are in this paper?"
        print(f"🔍 Query: {query}")
        
        results = multimodal_retrieve(
            session_id=session_id,
            question=query,
            store=store,
            memory_manager=memory_manager,
            top_n=3
        )
        
        print(f"✅ Results:")
        print(f"   - Text chunks: {len(results.get('text_chunks', []))}")
        print(f"   - Figures: {len(results.get('figures', []))}")
        print(f"   - Tables: {len(results.get('tables', []))}")
        print(f"   - Memory context: {len(results.get('memory_context', []))}")
    except Exception as exc:
        print(f"❌ Retrieval failed: {exc}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ Quick test completed successfully!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)

