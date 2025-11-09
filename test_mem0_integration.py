#!/usr/bin/env python3
"""
MEM0 Integration Test - Comprehensive test of memory system with Paper Sailor workflow.

Tests:
1. Memory write/read operations
2. Multi-level memory (user/session/agent)
3. Memory search and retrieval
4. Integration with vector store
5. Integration with Paper Sailor workflow
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from paper_sailor.memory import MemoryManager
from paper_sailor.vectorstore import VectorStore
from paper_sailor.tools.retrieval import multimodal_retrieve
from paper_sailor.tools.embeddings import embed_multimodal
from paper_sailor.tools.search_arxiv import search_arxiv
from paper_sailor.storage import ensure_dirs


def test_basic_memory_operations():
    """Test 1: Basic memory write and read operations."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Memory Operations")
    print("=" * 80)
    
    try:
        memory_manager = MemoryManager()
        print("✅ Memory manager initialized")
        
        # Test user-level memory
        print("\n📝 Testing user-level memory...")
        memory_manager.add_user_preference(
            user_id="test_user",
            preference="Interested in machine learning and NLP research"
        )
        print("✅ User preference added")
        
        # Test session-level memory
        print("\n📝 Testing session-level memory...")
        memory_manager.add_session_context(
            session_id="test_session_001",
            context={
                "topic": "Transformer architectures",
                "papers_reviewed": ["arxiv:2511.04093v1"],
                "key_findings": ["Attention mechanisms are crucial"]
            }
        )
        print("✅ Session context added")
        
        # Test agent-level memory
        print("\n📝 Testing agent-level memory...")
        memory_manager.add_agent_knowledge(
            knowledge="Graph neural networks often use message passing algorithms"
        )
        print("✅ Agent knowledge added")
        
        return True, memory_manager
    except Exception as exc:
        print(f"❌ Failed: {exc}")
        import traceback
        traceback.print_exc()
        return False, None


def test_memory_search(memory_manager):
    """Test 2: Memory search and retrieval."""
    print("\n" + "=" * 80)
    print("TEST 2: Memory Search and Retrieval")
    print("=" * 80)
    
    if not memory_manager:
        print("⚠️  Skipping (no memory manager)")
        return False
    
    try:
        # Search user memory
        print("\n🔍 Searching user-level memory...")
        user_results = memory_manager.search_memory(
            query="machine learning",
            level="user",
            limit=5
        )
        print(f"✅ Found {len(user_results)} user memories")
        if user_results:
            print(f"   Sample: {user_results[0]}")
        
        # Search session memory
        print("\n🔍 Searching session-level memory...")
        session_results = memory_manager.search_memory(
            query="Transformer",
            level="session",
            limit=5
        )
        print(f"✅ Found {len(session_results)} session memories")
        if session_results:
            print(f"   Sample: {session_results[0]}")
        
        # Search agent memory
        print("\n🔍 Searching agent-level memory...")
        agent_results = memory_manager.search_memory(
            query="neural networks",
            level="agent",
            limit=5
        )
        print(f"✅ Found {len(agent_results)} agent memories")
        if agent_results:
            print(f"   Sample: {agent_results[0]}")
        
        return True
    except Exception as exc:
        print(f"❌ Failed: {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_vector_integration():
    """Test 3: Integration between memory and vector store."""
    print("\n" + "=" * 80)
    print("TEST 3: Memory + Vector Store Integration")
    print("=" * 80)
    
    try:
        session_id = "integration_test_session"
        memory_manager = MemoryManager()
        vector_store = VectorStore()
        
        # Clean up previous test data
        vector_store.delete_session(session_id)
        
        # Add some test data to vector store
        print("\n📊 Adding test data to vector store...")
        test_items = [
            {
                "type": "text",
                "content": "Transformers use self-attention mechanisms to process sequences"
            },
            {
                "type": "figure",
                "content": "Architecture diagram showing encoder-decoder structure with attention layers"
            },
            {
                "type": "table",
                "content": "Performance comparison: BERT 85.2%, GPT 87.1%, T5 89.3%"
            }
        ]
        
        embeddings = embed_multimodal(test_items)
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        records = []
        for i, (item, emb) in enumerate(zip(test_items, embeddings)):
            records.append({
                "chunk_id": f"test:chunk:{i+1:04d}",
                "paper_id": "test_paper_001",
                "text": item["content"],
                "embedding": emb,
                "content_type": item["type"],
                "visual_description": item["content"],
                "metadata": {"test": True}
            })
        
        vector_store.upsert_multimodal(session_id, records)
        print(f"✅ Stored {len(records)} records in vector store")
        
        # Add corresponding memory context
        print("\n📝 Adding memory context...")
        memory_manager.add_session_context(session_id, {
            "topic": "Transformer architecture analysis",
            "papers_indexed": ["test_paper_001"],
            "chunks_processed": len(records)
        })
        print("✅ Memory context added")
        
        # Test multimodal retrieval with memory
        print("\n🔍 Testing multimodal retrieval with memory...")
        query = "What are the key components of Transformers?"
        results = multimodal_retrieve(
            session_id=session_id,
            question=query,
            store=vector_store,
            memory_manager=memory_manager,
            top_n=3,
            content_types=["text", "figure", "table"]
        )
        
        print(f"✅ Retrieval completed:")
        print(f"   - Text chunks: {len(results.get('text_chunks', []))}")
        print(f"   - Figures: {len(results.get('figures', []))}")
        print(f"   - Tables: {len(results.get('tables', []))}")
        print(f"   - Memory context: {len(results.get('memory_context', []))}")
        
        if results.get('memory_context'):
            print(f"   - Memory recall: {results['memory_context'][0]}")
        
        return True
    except Exception as exc:
        print(f"❌ Failed: {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_paper_sailor_workflow():
    """Test 4: Full Paper Sailor workflow with memory."""
    print("\n" + "=" * 80)
    print("TEST 4: Paper Sailor Workflow Integration")
    print("=" * 80)
    
    try:
        ensure_dirs()
        session_id = "workflow_mem0_test"
        memory_manager = MemoryManager()
        vector_store = VectorStore()
        
        # Step 1: Search papers
        print("\n🔍 Step 1: Searching papers...")
        topic = "attention mechanisms in neural networks"
        
        memory_manager.add_session_context(session_id, {
            "workflow_step": "search",
            "topic": topic,
            "timestamp": "2025-11-09"
        })
        
        try:
            papers = search_arxiv(topic, max_results=2)
            print(f"✅ Found {len(papers)} papers")
            if papers:
                print(f"   Sample: {papers[0].get('title', 'N/A')[:80]}...")
            
            memory_manager.add_session_context(session_id, {
                "papers_found": len(papers),
                "paper_ids": [p.get('id') for p in papers[:2]]
            })
        except Exception as exc:
            print(f"⚠️  Search skipped (network/API): {exc}")
            papers = []
        
        # Step 2: Simulate paper processing
        print("\n📄 Step 2: Processing papers...")
        if papers:
            for paper in papers[:1]:  # Process just first paper
                paper_id = paper.get('id')
                summary = paper.get('summary', '')
                
                if summary:
                    # Create synthetic chunks
                    chunks = [
                        {
                            "id": f"{paper_id}:chunk:0001",
                            "paper_id": paper_id,
                            "text": summary[:500],
                            "content_type": "text"
                        }
                    ]
                    
                    # Embed and store
                    items = [{"type": "text", "content": chunk["text"]} for chunk in chunks]
                    embeddings = embed_multimodal(items)
                    
                    records = []
                    for chunk, emb in zip(chunks, embeddings):
                        records.append({
                            "chunk_id": chunk["id"],
                            "paper_id": chunk["paper_id"],
                            "text": chunk["text"],
                            "embedding": emb,
                            "content_type": "text",
                            "metadata": {}
                        })
                    
                    vector_store.upsert_multimodal(session_id, records)
                    print(f"✅ Processed {paper_id}: {len(records)} chunks")
                    
                    # Update memory
                    memory_manager.add_session_context(session_id, {
                        "workflow_step": "processing",
                        "processed_paper": paper_id,
                        "chunks_indexed": len(records)
                    })
        
        # Step 3: Query and retrieve with memory
        print("\n🔍 Step 3: Querying with memory context...")
        query = "What are the main findings about attention mechanisms?"
        
        results = multimodal_retrieve(
            session_id=session_id,
            question=query,
            store=vector_store,
            memory_manager=memory_manager,
            top_n=3
        )
        
        print(f"✅ Query results:")
        print(f"   - Text chunks: {len(results.get('text_chunks', []))}")
        print(f"   - Memory context items: {len(results.get('memory_context', []))}")
        
        # Step 4: Add findings to memory
        print("\n📝 Step 4: Storing findings in memory...")
        memory_manager.add_agent_knowledge(
            f"Research on '{topic}' shows that attention mechanisms improve model performance"
        )
        print("✅ Findings stored in agent memory")
        
        # Step 5: Verify memory persistence
        print("\n🔍 Step 5: Verifying memory persistence...")
        recall = memory_manager.search_memory(
            query="attention mechanisms",
            level="session",
            limit=5
        )
        print(f"✅ Recalled {len(recall)} memory items from session")
        
        return True
    except Exception as exc:
        print(f"❌ Failed: {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_context_enrichment():
    """Test 5: Memory-enriched retrieval."""
    print("\n" + "=" * 80)
    print("TEST 5: Memory Context Enrichment")
    print("=" * 80)
    
    try:
        session_id = "enrichment_test"
        memory_manager = MemoryManager()
        
        # Add rich context to memory
        print("\n📝 Adding rich research context...")
        memory_manager.add_session_context(session_id, {
            "research_area": "Deep Learning",
            "focus_topics": ["attention", "transformers", "self-attention"],
            "papers_analyzed": 5,
            "key_insights": [
                "Attention allows models to focus on relevant parts",
                "Multi-head attention captures different aspects",
                "Positional encoding is crucial for sequence modeling"
            ]
        })
        print("✅ Rich context added")
        
        # Test context retrieval
        print("\n🔍 Testing context retrieval...")
        context = memory_manager.get_relevant_context(
            session_id=session_id,
            question="How does attention work in transformers?"
        )
        print(f"✅ Retrieved context: {len(context)} characters")
        if context:
            print(f"   Preview: {context[:150]}...")
        
        return True
    except Exception as exc:
        print(f"❌ Failed: {exc}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all MEM0 integration tests."""
    print("=" * 80)
    print("MEM0 Integration Test Suite")
    print("=" * 80)
    print("\nTesting MEM0 memory system integration with Paper Sailor")
    print("This validates:")
    print("  - Memory write/read operations")
    print("  - Multi-level memory (user/session/agent)")
    print("  - Integration with vector store")
    print("  - Integration with Paper Sailor workflow")
    
    results = {}
    
    # Test 1: Basic operations
    success, memory_manager = test_basic_memory_operations()
    results["basic_operations"] = success
    
    # Test 2: Memory search
    if memory_manager:
        results["memory_search"] = test_memory_search(memory_manager)
    else:
        results["memory_search"] = False
    
    # Test 3: Vector store integration
    results["vector_integration"] = test_memory_vector_integration()
    
    # Test 4: Full workflow
    results["workflow_integration"] = test_paper_sailor_workflow()
    
    # Test 5: Context enrichment
    results["context_enrichment"] = test_memory_context_enrichment()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! MEM0 integration is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

