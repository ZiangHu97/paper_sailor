from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PlannerPrompts:
    """Prompts for the planner agent."""
    
    system: str = """
You are the Planner for the Paper Sailor research agent. Your job is to guide a
multi-step exploration of scientific papers for the given topic. At each turn
you will see the current memory and a summary of the previous executor result.

Always respond with a single JSON object. The JSON must contain:

{
  "action": string,            # one of: search, read, summarize, reflect, critique, finish
  "queries": [ ... ],          # required when action == "search"
  "papers": [ ... ],           # required when action == "read"
  "focus": [ ... ],            # required when action == "summarize"
  "notes": string,             # brief intent rationale
  "todo": [
      {"title": string, "status": "todo" | "doing" | "done"}
  ]
}

Actions:
- search: Generate 1-3 search queries to find papers. Prefer arXiv field syntax
  (e.g., "all:graph AND all:molecules"), otherwise plain keywords.
- read: Choose paper ids from known papers to download and chunk.
- summarize: List focus questions or themes to synthesize using available chunks.
- reflect: Pause to assess progress, identify gaps, and adjust strategy.
- critique: Evaluate the quality and relevance of current findings.
- finish: End the session when major questions are answered or budget exhausted.

Rules:
- Use the todo list to track medium-term subgoals. Update statuses explicitly.
- Reflect periodically (every 3-4 steps) to ensure you're on track.
- Critique your findings to identify weak evidence or missing perspectives.
- Finish only when you have comprehensive coverage or reach the round limit.
- Provide a short summary in notes when finishing.
"""
    
    reflection: str = """
You are reflecting on the research progress so far. Review the state and answer:

1. What have we learned? (key findings summary)
2. What gaps remain? (missing topics, weak evidence)
3. Are we on track? (alignment with original topic)
4. What should we do next? (strategic recommendation)

Provide a structured reflection in JSON:
{
  "summary": "Brief summary of progress",
  "findings_quality": "Assessment of current findings (strong/moderate/weak)",
  "coverage_gaps": ["gap1", "gap2", ...],
  "strategy_adjustment": "Recommended next steps",
  "confidence": 0.0-1.0
}
"""
    
    critique: str = """
You are critiquing the quality of current research findings. Evaluate:

1. Evidence strength: Are findings well-supported by papers?
2. Relevance: Do findings address the original research topic?
3. Diversity: Have we explored multiple perspectives/approaches?
4. Depth: Is the analysis superficial or thorough?

Provide critique in JSON:
{
  "overall_quality": "excellent/good/fair/poor",
  "strengths": ["strength1", "strength2", ...],
  "weaknesses": ["weakness1", "weakness2", ...],
  "recommendations": ["action1", "action2", ...],
  "confidence": 0.0-1.0
}
"""
    
    temperature: float = 0.2
    max_tokens: int = 1600
    
    def render_system(self, **kwargs: Any) -> str:
        """Render system prompt with optional variable substitution."""
        return self.system.format(**kwargs) if kwargs else self.system
    
    def render_reflection(self, **kwargs: Any) -> str:
        """Render reflection prompt with optional variable substitution."""
        return self.reflection.format(**kwargs) if kwargs else self.reflection
    
    def render_critique(self, **kwargs: Any) -> str:
        """Render critique prompt with optional variable substitution."""
        return self.critique.format(**kwargs) if kwargs else self.critique


@dataclass
class AgentPrompts:
    """Prompts for the agent's question and idea generation."""
    
    question_generation: str = """
You draft investigative research questions for a literature review.
Return JSON {{"questions": [..]}} with concise, topic-specific questions (max one sentence each).

Topic: {topic}
{context}
Provide up to {max_questions} focused questions.
"""
    
    idea_generation: str = """
You are a research strategist. Based on findings and key papers, propose actionable next steps.
Return JSON {{"ideas": [{{"title":...,"motivation":...,"method":...,"eval":...,"risks":...,"refs": [...]}}]}}

Topic: {topic}
Key papers:
{papers}
Findings summary:
{findings}
Propose up to {max_ideas} reflective ideas with ref IDs pulled from the key papers list.
"""
    
    answer_synthesis: str = """
You are a research assistant. Write concise answers grounded in the provided excerpts. Cite using paper ids.

Question: {question}
Excerpts:
{context}
"""
    
    temperature: float = 0.3
    max_tokens: int = 800
    
    def render_question_generation(self, topic: str, context: str = "", max_questions: int = 4) -> str:
        """Render question generation prompt."""
        return self.question_generation.format(
            topic=topic,
            context=context,
            max_questions=max_questions
        )
    
    def render_idea_generation(self, topic: str, papers: str, findings: str, max_ideas: int = 3) -> str:
        """Render idea generation prompt."""
        return self.idea_generation.format(
            topic=topic,
            papers=papers,
            findings=findings,
            max_ideas=max_ideas
        )
    
    def render_answer_synthesis(self, question: str, context: str) -> str:
        """Render answer synthesis prompt."""
        return self.answer_synthesis.format(
            question=question,
            context=context
        )


@dataclass
class ToolPrompts:
    """Prompts for tool-related operations (future: code generation, analysis)."""
    
    code_generation: str = """
Generate Python code to verify the following research idea:

Idea: {idea_title}
Hypothesis: {hypothesis}
Context: {context}

Requirements:
- Use only standard scientific libraries (numpy, scipy, matplotlib, pandas)
- Include assertions to validate key claims
- Add comments explaining the logic
- Return results as a dictionary

Generate executable Python code that tests this idea.
"""
    
    code_analysis: str = """
Analyze the following code execution results:

Code: {code}
Stdout: {stdout}
Stderr: {stderr}
Success: {success}

Provide a brief interpretation:
1. What was tested?
2. Did it succeed or fail?
3. What do the results tell us?
4. Confidence in the findings (0.0-1.0)
"""
    
    temperature: float = 0.2
    max_tokens: int = 1200
    
    def render_code_generation(self, idea_title: str, hypothesis: str, context: str) -> str:
        """Render code generation prompt."""
        return self.code_generation.format(
            idea_title=idea_title,
            hypothesis=hypothesis,
            context=context
        )
    
    def render_code_analysis(self, code: str, stdout: str, stderr: str, success: bool) -> str:
        """Render code analysis prompt."""
        return self.code_analysis.format(
            code=code,
            stdout=stdout,
            stderr=stderr,
            success=success
        )


# Global instances with defaults
_planner_prompts: Optional[PlannerPrompts] = None
_agent_prompts: Optional[AgentPrompts] = None
_tool_prompts: Optional[ToolPrompts] = None


def get_planner_prompts() -> PlannerPrompts:
    """Get planner prompts (with config overrides if available)."""
    global _planner_prompts
    if _planner_prompts is None:
        from .config import get_prompt_overrides
        overrides = get_prompt_overrides("planner")
        _planner_prompts = PlannerPrompts(**overrides)
    return _planner_prompts


def get_agent_prompts() -> AgentPrompts:
    """Get agent prompts (with config overrides if available)."""
    global _agent_prompts
    if _agent_prompts is None:
        from .config import get_prompt_overrides
        overrides = get_prompt_overrides("agent")
        _agent_prompts = AgentPrompts(**overrides)
    return _agent_prompts


def get_tool_prompts() -> ToolPrompts:
    """Get tool prompts (with config overrides if available)."""
    global _tool_prompts
    if _tool_prompts is None:
        from .config import get_prompt_overrides
        overrides = get_prompt_overrides("tool")
        _tool_prompts = ToolPrompts(**overrides)
    return _tool_prompts


def reset_prompts() -> None:
    """Reset cached prompts (useful for testing or config reload)."""
    global _planner_prompts, _agent_prompts, _tool_prompts
    _planner_prompts = None
    _agent_prompts = None
    _tool_prompts = None

