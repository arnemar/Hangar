"""Built-in skills (slash commands) for the agent chat.

Skills are prompt templates invoked by typing /skill-name in the chat input.
The frontend appends them to the user's message; the backend just serves them.
"""

BUILTIN_SKILLS: list[dict] = [
    {
        "name": "review",
        "description": "Review code for bugs, security issues, and improvements",
        "prompt": "Please review the following code for bugs, security issues, performance problems, and style improvements. Be specific and actionable:\n\n",
        "icon": "🔍",
    },
    {
        "name": "explain",
        "description": "Explain what code does in plain language",
        "prompt": "Please explain what this code does, step by step, in plain language. Focus on the key logic and any non-obvious parts:\n\n",
        "icon": "💡",
    },
    {
        "name": "fix",
        "description": "Diagnose and fix a bug",
        "prompt": "There is a bug in this code. Please diagnose the root cause and provide a working fix with explanation:\n\n",
        "icon": "🐛",
    },
    {
        "name": "test",
        "description": "Write tests for code",
        "prompt": "Write comprehensive tests for the following code. Include happy path, edge cases, and error conditions:\n\n",
        "icon": "✅",
    },
    {
        "name": "refactor",
        "description": "Refactor code for clarity and maintainability",
        "prompt": "Refactor this code to improve readability, reduce complexity, and follow best practices. Explain the key changes you made:\n\n",
        "icon": "♻️",
    },
    {
        "name": "docs",
        "description": "Add documentation and docstrings",
        "prompt": "Add clear documentation, docstrings, and inline comments where needed to this code:\n\n",
        "icon": "📝",
    },
    {
        "name": "commit",
        "description": "Write a git commit message for staged changes",
        "prompt": "Look at the git diff --staged output and write a clear, concise commit message following conventional commits format (type: short description). First run git diff --staged to see what's staged.",
        "icon": "📦",
    },
    {
        "name": "optimize",
        "description": "Optimize code for performance",
        "prompt": "Analyze this code for performance bottlenecks and suggest concrete optimizations with expected impact:\n\n",
        "icon": "⚡",
    },
    {
        "name": "security",
        "description": "Audit code for security vulnerabilities",
        "prompt": "Audit this code for security vulnerabilities including injection, auth issues, data exposure, and other OWASP Top 10 risks. Be specific about each finding:\n\n",
        "icon": "🔒",
    },
    {
        "name": "types",
        "description": "Add type annotations",
        "prompt": "Add proper type annotations to this code. Include all function signatures, variable types where useful, and generics where appropriate:\n\n",
        "icon": "📐",
    },
]


def get_all_skills(custom: list[dict] | None = None) -> list[dict]:
    """Return built-in skills merged with any user-defined custom skills."""
    builtin_names = {s["name"] for s in BUILTIN_SKILLS}
    result = [{"_builtin": True, **s} for s in BUILTIN_SKILLS]
    for s in (custom or []):
        if s["name"] not in builtin_names:
            result.append({"_builtin": False, **s})
    return result
