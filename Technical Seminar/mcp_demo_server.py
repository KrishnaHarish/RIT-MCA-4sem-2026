"""Real MCP server entry point for the technical seminar demo.

This uses the official `mcp` Python SDK with FastMCP and serves four demo roles:
- filesystem
- github
- database
- knowledge_base

Run one server instance per role:
    /workspaces/RIT-MCA-4sem-2026/.venv/bin/python "Technical Seminar/mcp_demo_server.py" --role filesystem
    /workspaces/RIT-MCA-4sem-2026/.venv/bin/python "Technical Seminar/mcp_demo_server.py" --role github
    /workspaces/RIT-MCA-4sem-2026/.venv/bin/python "Technical Seminar/mcp_demo_server.py" --role database
    /workspaces/RIT-MCA-4sem-2026/.venv/bin/python "Technical Seminar/mcp_demo_server.py" --role knowledge_base
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


ROLE_NAMES = {"filesystem", "github", "database", "knowledge_base"}


def create_server(role: str) -> FastMCP:
    app = FastMCP(name=f"technical-seminar-{role}", log_level="WARNING")

    if role == "filesystem":
        @app.tool()
        def search_docs(query: str) -> dict[str, Any]:
            return {"results": [f"Matched document for: {query}"]}

        @app.tool()
        def summarize_doc(title: str) -> dict[str, Any]:
            return {"summary": f"Summary prepared for {title}."}

        @app.tool()
        def read_file(path: str) -> dict[str, Any]:
            return {"content": f"Contents of {path}."}

    elif role == "github":
        @app.tool()
        def repo_info(repo: str = "demo/repo", branch: str = "main") -> dict[str, Any]:
            return {"repo": repo, "branch": branch, "default_branch": "main"}

        @app.tool()
        def list_commits(repo: str = "demo/repo") -> dict[str, Any]:
            seed = sum(ord(char) for char in repo)
            commits = [f"{seed + index:06x}"[-6:] for index in range(1, 4)]
            return {"commits": commits}

        @app.tool()
        def triage_issue(issue: int) -> dict[str, Any]:
            labels = ["bug", "triage"] if issue % 2 == 0 else ["question", "needs-review"]
            return {"issue": issue, "labels": labels, "priority": "high" if issue % 2 == 0 else "medium"}

        @app.tool()
        def open_pr(repo: str = "demo/repo", branch: str = "feature/demo", title: str = "Fix demo build") -> dict[str, Any]:
            seed = sum(ord(char) for char in branch)
            return {"repo": repo, "branch": branch, "title": title, "url": f"https://github.com/demo/repo/pull/{seed % 100 + 1}"}

    elif role == "database":
        @app.tool()
        def run_query(sql: str) -> dict[str, Any]:
            return {"rows": 12, "sample": [{"query": sql[:32], "count": 4}]}

        @app.tool()
        def generate_report(title: str) -> dict[str, Any]:
            return {"report": f"{title} generated."}

    elif role == "knowledge_base":
        @app.tool()
        def retrieve_policy(topic: str) -> dict[str, Any]:
            return {"policy": f"Policy guidance for {topic}: write actions require explicit user approval."}

        @app.tool()
        def answer_faq(question: str) -> dict[str, Any]:
            return {"answer": f"FAQ response for {question}."}

        @app.tool()
        def fetch_guidance(topic: str) -> dict[str, Any]:
            return {"guidance": f"Guidance for {topic}: use structured logs and least privilege."}

    else:
        raise ValueError(f"Unsupported role: {role}")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Technical seminar MCP demo server")
    parser.add_argument("--role", required=True, choices=sorted(ROLE_NAMES))
    args = parser.parse_args()

    server = create_server(args.role)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
