#!/usr/bin/env python3
"""
Simple script to verify MEM0 API calls are actually happening.
Check your MEM0 dashboard for these operations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from paper_sailor.memory import MemoryManager

print("=" * 80)
print("MEM0 API Call Verification")
print("=" * 80)
print("\n🔍 Check your MEM0 dashboard at https://app.mem0.ai/")
print("You should see these API calls:\n")

# Initialize
print("\n1️⃣  Initializing MEM0 SDK...")
memory = MemoryManager()

if not memory.use_mem0:
    print("❌ MEM0 SDK not initialized. Using local storage.")
    print("   Please check your config.toml has mem0.api_key set.")
    sys.exit(1)

print(f"✅ MEM0 SDK initialized: {type(memory.mem0_client)}")

# Test 1: Add user preference
print("\n2️⃣  Adding user preference...")
memory.add_user_preference(
    user_id="test_user_verify",
    preference="I am interested in deep learning and computer vision research"
)
print("✅ Check dashboard for: user_id='test_user_verify'")

# Test 2: Add session context
print("\n3️⃣  Adding session context...")
memory.add_session_context(
    session_id="verify_session_001",
    context={
        "topic": "Transformer architectures for NLP",
        "papers_reviewed": ["arxiv:1706.03762"],
        "timestamp": "2025-11-09"
    }
)
print("✅ Check dashboard for: user_id='session_verify_session_001'")

# Test 3: Add agent knowledge  
print("\n4️⃣  Adding agent knowledge...")
memory.add_agent_knowledge(
    "Self-attention mechanisms allow models to weigh the importance of different parts of the input"
)
print("✅ Check dashboard for: user_id='agent_global'")

# Test 4: Search memories
print("\n5️⃣  Searching memories...")
try:
    results = memory.search_memory(
        query="deep learning",
        level="user",
        limit=5
    )
    print(f"✅ Search completed, found {len(results)} results")
except Exception as exc:
    print(f"⚠️  Search failed: {exc}")

print("\n" + "=" * 80)
print("✅ VERIFICATION COMPLETE")
print("=" * 80)
print("\n📊 Go to https://app.mem0.ai/dashboard and look for:")
print("   - 3 .add() calls (user, session, agent)")
print("   - 1 .search() call (if successful)")
print("\nIf you see these calls, MEM0 integration is working! 🎉")

