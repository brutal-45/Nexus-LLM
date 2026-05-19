"""Code agent for code generation, execution, debugging, and review.

Provides a specialized agent that can generate code, execute it in
a sandbox, debug errors, and review code for quality and security.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.agents.base import Agent, AgentAction, AgentConfig, AgentObservation
from nexus_llm.agents.executor import ActionExecutor
from nexus_llm.agents.memory import AgentMemory, ShortTermMemory
from nexus_llm.agents.tools import CodeRunTool, FileReadTool, FileWriteTool, Tool

logger = logging.getLogger(__name__)


class CodeAgent(Agent):
    """Agent specialized in code generation, execution, and debugging.

    Supports a write-run-debug cycle: generates code, executes it
    in a sandbox, detects errors, and iteratively fixes issues.
    Also provides code review capabilities.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[Dict[str, Tool]] = None,
        memory: Optional[AgentMemory] = None,
        llm_fn: Optional[Callable] = None,
        executor: Optional[ActionExecutor] = None,
        max_debug_iterations: int = 3,
    ):
        config = config or AgentConfig(
            name="CodeAgent",
            description="An agent that generates, executes, debugs, and reviews code",
            max_iterations=10,
        )
        super().__init__(config=config, tools=tools, memory=memory, llm_fn=llm_fn)

        # Ensure essential tools are available
        if "code_run" not in self.tools:
            self.add_tool(CodeRunTool())
        if "file_read" not in self.tools:
            self.add_tool(FileReadTool())
        if "file_write" not in self.tools:
            self.add_tool(FileWriteTool())

        self.executor = executor or ActionExecutor(tools=self.tools)
        self.max_debug_iterations = max_debug_iterations
        self._debug_count = 0
        self._current_code: Optional[str] = None

    def generate_code(self, specification: str, language: str = "python") -> str:
        """Generate code from a specification.

        Args:
            specification: Natural language description of what the code should do.
            language: Programming language (currently only Python supported).

        Returns:
            Generated code string.
        """
        if self.llm_fn:
            prompt = (
                f"Generate {language} code for the following specification:\n\n"
                f"{specification}\n\n"
                f"Requirements:\n"
                f"- Write clean, well-structured code\n"
                f"- Include error handling\n"
                f"- Add comments for complex logic\n"
                f"- Return only the code, no explanations\n\n"
                f"Code:"
            )
            try:
                response = self.llm_fn(prompt)
                code = self._extract_code(response)
                self._current_code = code
                return code
            except Exception as e:
                logger.error("Code generation failed: %s", e)
                return f"# Error generating code: {e}"

        # Fallback: simple template-based code generation
        return self._template_generate(specification, language)

    def execute_code(self, code: str, timeout: int = 30) -> AgentObservation:
        """Execute code in a sandbox.

        Args:
            code: Python code to execute.
            timeout: Execution timeout in seconds.

        Returns:
            AgentObservation with execution results.
        """
        self._current_code = code
        result = self.executor.execute("code_run", code=code, timeout=timeout)

        obs = AgentObservation(
            action=AgentAction(
                action_type="tool_call",
                tool_name="code_run",
                tool_args={"code": code[:200]},
            ),
            result=result,
            observation_text=result.output if result.success else f"Error: {result.error}",
            success=result.success,
        )
        self.observation_history.append(obs)
        return obs

    def debug(self, code: str, error: str) -> str:
        """Debug code given an error message.

        Args:
            code: The code that produced an error.
            error: The error message.

        Returns:
            Fixed code string.
        """
        if self._debug_count >= self.max_debug_iterations:
            logger.warning("Max debug iterations reached (%d).", self.max_debug_iterations)
            return code

        self._debug_count += 1

        if self.llm_fn:
            prompt = (
                f"The following Python code produced an error. Fix it.\n\n"
                f"Code:\n```python\n{code}\n```\n\n"
                f"Error:\n```\n{error}\n```\n\n"
                f"Return only the fixed code, no explanations."
            )
            try:
                response = self.llm_fn(prompt)
                fixed = self._extract_code(response)
                return fixed
            except Exception as e:
                logger.error("LLM debugging failed: %s", e)

        # Rule-based debugging fallback
        return self._rule_debug(code, error)

    def review_code(self, code: str) -> Dict[str, Any]:
        """Review code for quality, security, and best practices.

        Args:
            code: Python code to review.

        Returns:
            Review results dictionary.
        """
        issues: List[Dict[str, str]] = []
        suggestions: List[str] = []
        score = 100  # Start at 100, deduct for issues

        # Security checks
        dangerous_patterns = [
            (r"eval\s*\(", "Use of eval() is a security risk", "high"),
            (r"exec\s*\(", "Use of exec() is a security risk", "high"),
            (r"__import__\s*\(", "Dynamic imports can be dangerous", "medium"),
            (r"subprocess\.call", "Subprocess call without shell=False may be unsafe", "medium"),
            (r"os\.system\s*\(", "os.system() is unsafe, use subprocess instead", "high"),
            (r"open\s*\([^)]*['\"]w", "File write operations should be validated", "low"),
        ]

        for pattern, message, severity in dangerous_patterns:
            matches = re.findall(pattern, code)
            if matches:
                issues.append({"type": "security", "message": message, "severity": severity})
                if severity == "high":
                    score -= 20
                elif severity == "medium":
                    score -= 10
                else:
                    score -= 5

        # Quality checks
        lines = code.split("\n")
        non_empty_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]

        # Check for comments
        comment_lines = [l for l in lines if l.strip().startswith("#")]
        if non_empty_lines and len(comment_lines) / max(len(non_empty_lines), 1) < 0.1:
            suggestions.append("Add more comments to explain complex logic")
            score -= 5

        # Check for very long functions
        function_starts = [i for i, l in enumerate(lines) if re.match(r"^\s*def\s+", l)]
        for idx in range(len(function_starts)):
            start = function_starts[idx]
            end = function_starts[idx + 1] if idx + 1 < len(function_starts) else len(lines)
            if end - start > 50:
                issues.append({
                    "type": "quality",
                    "message": f"Function at line {start + 1} is too long ({end - start} lines). Consider breaking it up.",
                    "severity": "medium",
                })
                score -= 10

        # Check for bare except clauses
        if re.search(r"except\s*:", code):
            issues.append({
                "type": "quality",
                "message": "Bare 'except:' clause catches all exceptions. Use specific exception types.",
                "severity": "medium",
            })
            score -= 10

        # Check for hardcoded secrets
        if re.search(r"(?:password|secret|api_key|token)\s*=\s*['\"][^'\"]+['\"]", code, re.IGNORECASE):
            issues.append({
                "type": "security",
                "message": "Hardcoded secret detected. Use environment variables instead.",
                "severity": "high",
            })
            score -= 20

        # Check for type hints (Python 3+)
        if not re.search(r"def\s+\w+\s*\([^)]*:\s*\w+", code):
            suggestions.append("Consider adding type hints to function signatures")

        # Check for docstrings
        if not re.search(r'"""', code) and not re.search(r"'''", code):
            suggestions.append("Add docstrings to document functions and classes")
            score -= 5

        score = max(0, score)

        return {
            "score": score,
            "issues": issues,
            "suggestions": suggestions,
            "summary": self._generate_review_summary(score, issues, suggestions),
        }

    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Optional[AgentAction]:
        """Decide action based on current state."""
        # Check if last execution had errors
        if self.observation_history:
            last_obs = self.observation_history[-1]
            if not last_obs.success and self._current_code and self._debug_count < self.max_debug_iterations:
                # Debug the code
                error_msg = last_obs.observation_text
                fixed_code = self.debug(self._current_code, error_msg)
                return AgentAction(
                    action_type="tool_call",
                    tool_name="code_run",
                    tool_args={"code": fixed_code},
                    thought=f"Debugging code (attempt {self._debug_count}/{self.max_debug_iterations})",
                )

            if last_obs.success:
                # Task complete
                return AgentAction(
                    action_type="respond",
                    response=self._format_success_response(),
                    thought="Code executed successfully.",
                )

        # First iteration: generate and run code
        code = self.generate_code(task)
        return AgentAction(
            action_type="tool_call",
            tool_name="code_run",
            tool_args={"code": code},
            thought="Generating and executing initial code.",
        )

    def act(self, action: AgentAction) -> AgentObservation:
        """Execute an action."""
        if action.action_type == "tool_call" and action.tool_name == "code_run":
            code = action.tool_args.get("code", "") if action.tool_args else ""
            result = self.executor.execute("code_run", code=code)
            self._current_code = code
            return AgentObservation(
                action=action,
                result=result,
                observation_text=result.output if result.success else f"Error: {result.error}",
                success=result.success,
            )
        return super().act(action)

    def _extract_code(self, response: str) -> str:
        """Extract code from an LLM response, handling markdown code blocks."""
        # Check for markdown code blocks
        code_block = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        # Check for indented code blocks
        lines = response.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if re.match(r"^\s*(def |class |import |from |if |for |while |try |with )", line):
                in_code = True
            if in_code:
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines)

        return response.strip()

    def _rule_debug(self, code: str, error: str) -> str:
        """Rule-based debugging for common errors."""
        fixed = code

        # Fix common indentation errors
        if "IndentationError" in error or "unexpected indent" in error:
            lines = fixed.split("\n")
            # Re-indent using 4-space standard
            indent_level = 0
            new_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    new_lines.append("")
                    continue
                if stripped.startswith(("return", "break", "continue", "pass", "raise")):
                    indent_level = max(0, indent_level - 1)
                new_lines.append("    " * indent_level + stripped)
                if stripped.endswith(":"):
                    indent_level += 1
                elif stripped.startswith(("return", "break", "continue", "pass")):
                    indent_level = max(0, indent_level - 1)
            fixed = "\n".join(new_lines)

        # Fix missing colons after control statements
        for keyword in ["if", "elif", "else", "for", "while", "def", "class", "try", "except", "finally", "with"]:
            pattern = rf"^(\s*{keyword}\s+.+[^:])$"
            fixed = re.sub(pattern, r"\1:", fixed, flags=re.MULTILINE)

        # Fix undefined name errors by adding common imports
        import_fixes = {
            "math": "import math",
            "os": "import os",
            "sys": "import sys",
            "json": "import json",
            "re": "import re",
            "datetime": "import datetime",
            "collections": "import collections",
            "itertools": "import itertools",
        }
        for module, import_stmt in import_fixes.items():
            if f"NameError: name '{module}'" in error or f"name '{module}' is not defined" in error:
                if import_stmt not in fixed:
                    fixed = import_stmt + "\n" + fixed

        # Fix print statement vs function (Python 3)
        fixed = re.sub(r"print\s+([^(].*?)(\n|$)", r"print(\1)\2", fixed)

        return fixed

    def _template_generate(self, specification: str, language: str) -> str:
        """Template-based code generation fallback."""
        return (
            f'# Generated {language} code for: {specification}\n'
            f'def solution():\n'
            f'    """Solution for: {specification}"""\n'
            f'    # TODO: Implement the solution\n'
            f'    pass\n\n'
            f'if __name__ == "__main__":\n'
            f'    result = solution()\n'
            f'    print(result)\n'
        )

    def _format_success_response(self) -> str:
        """Format a success response with code and output."""
        parts = []
        if self._current_code:
            parts.append("Code:")
            parts.append(f"```python\n{self._current_code}\n```")

        for obs in self.observation_history:
            if obs.success and obs.observation_text:
                parts.append(f"\nOutput:\n{obs.observation_text}")

        return "\n".join(parts)

    def reset(self) -> None:
        """Reset the code agent."""
        super().reset()
        self._debug_count = 0
        self._current_code = None
