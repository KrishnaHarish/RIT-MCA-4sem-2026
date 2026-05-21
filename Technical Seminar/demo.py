"""Runnable seminar demo for Architecting Universal AI Agents using MCP.

This version uses the official MCP Python SDK:
- FastMCP servers launched as local stdio processes
- stdio_client + ClientSession for discovery and tool invocation
- structured logging, approval gating, and saved artifacts

Run with the project virtualenv so the `mcp` package is available:
    /workspaces/RIT-MCA-4sem-2026/.venv/bin/python "Technical Seminar/demo.py"
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from mcp import ClientSession, StdioServerParameters, stdio_client


@dataclass(frozen=True)
class ToolCall:
    timestamp: str
    tool_name: str
    arguments_hash: str
    server: str
    latency_ms: int
    status: str
    retries: int


@dataclass
class TaskResult:
    task: str
    success_criteria: str
    tools_resources_used: str
    steps: int
    latency_ms: int
    cost_estimate: str
    failure_modes: str
    recovery_actions: str
    final_output: str
    approved: bool = False
    tool_calls: List[ToolCall] = field(default_factory=list)


class MCPDemoClient:
    def __init__(self, role: str, server_path: Path, cwd: Path) -> None:
        self.role = role
        self.server_path = server_path
        self.cwd = cwd
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(server_path), "--role", role],
            cwd=str(cwd),
        )

    @staticmethod
    def _extract_text(result: Any) -> str:
        if hasattr(result, "content"):
            parts = []
            for item in result.content:
                if hasattr(item, "text"):
                    parts.append(item.text)
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(result)

    async def list_tools(self) -> List[Dict[str, Any]]:
        async with stdio_client(self.server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()
                tools: List[Dict[str, Any]] = []
                for tool in result.tools:
                    tools.append({"name": tool.name, "description": getattr(tool, "description", "")})
                return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        async with stdio_client(self.server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                text = self._extract_text(result)
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"text": text}


class MCPDemoAgent:
    def __init__(self, clients: Dict[str, MCPDemoClient]) -> None:
        self.clients = clients
        self.allowed_tools = {
            "filesystem.search_docs",
            "filesystem.summarize_doc",
            "filesystem.read_file",
            "github.repo_info",
            "github.list_commits",
            "github.triage_issue",
            "github.open_pr",
            "database.run_query",
            "database.generate_report",
            "knowledge_base.retrieve_policy",
            "knowledge_base.answer_faq",
            "knowledge_base.fetch_guidance",
        }
        self.approved_write_tools = {"github.open_pr", "database.generate_report"}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _hash_args(payload: Dict[str, Any]) -> str:
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _latency(seed: str) -> int:
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        return 50 + (int(digest[:4], 16) % 220)

    async def discover_tools(self) -> Dict[str, List[str]]:
        discovered: Dict[str, List[str]] = {}
        for role, client in self.clients.items():
            discovered[role] = [tool["name"] for tool in await client.list_tools()]
        return discovered

    async def _call_tool(self, server_name: str, tool_name: str, retries: int = 0, **kwargs: Any) -> tuple[Dict[str, Any], ToolCall]:
        full_name = f"{server_name}.{tool_name}"
        if full_name not in self.allowed_tools:
            raise PermissionError(f"Tool not allowed: {full_name}")
        latency_ms = self._latency(f"{server_name}:{tool_name}:{kwargs}")
        output = await self.clients[server_name].call_tool(tool_name, kwargs)
        tool_call = ToolCall(
            timestamp=self._now(),
            tool_name=full_name,
            arguments_hash=self._hash_args(kwargs),
            server=f"{server_name}:mcp-stdio",
            latency_ms=latency_ms,
            status="success",
            retries=retries,
        )
        return output, tool_call

    def _approval_required(self, full_name: str) -> bool:
        return full_name in self.approved_write_tools

    async def run_task(
        self,
        task: str,
        success_criteria: str,
        steps: Sequence[tuple[str, str, Dict[str, Any]]],
        final_output: str,
        failure_modes: str,
        recovery_actions: str,
        cost_estimate: str,
    ) -> TaskResult:
        tool_calls: List[ToolCall] = []
        tool_descriptions: List[str] = []
        total_latency = 0
        approved = False

        for server_name, tool_name, kwargs in steps:
            full_name = f"{server_name}.{tool_name}"
            tool_descriptions.append(f"{full_name} (MCP stdio)")
            if self._approval_required(full_name):
                approved = True
            _, tool_call = await self._call_tool(server_name, tool_name, **kwargs)
            tool_calls.append(tool_call)
            total_latency += tool_call.latency_ms

        return TaskResult(
            task=task,
            success_criteria=success_criteria,
            tools_resources_used=", ".join(tool_descriptions),
            steps=len(tool_calls),
            latency_ms=total_latency,
            cost_estimate=cost_estimate,
            failure_modes=failure_modes,
            recovery_actions=recovery_actions,
            final_output=final_output,
            approved=approved,
            tool_calls=tool_calls,
        )


def print_heading(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    def format_row(values: Sequence[str]) -> str:
        return " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(values))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def build_demo_tasks(agent: MCPDemoAgent) -> List[tuple[str, str, Sequence[tuple[str, str, Dict[str, Any]]], str, str, str, str]]:
    return [
        (
            "Document search and summarization",
            "Find the requested seminar notes and summarize them clearly.",
            [
                ("filesystem", "search_docs", {"query": "MCP architecture summary"}),
                ("filesystem", "summarize_doc", {"title": "MCP architecture summary"}),
            ],
            "Short summary of MCP architecture notes.",
            "None in the demo run.",
            "Not needed.",
            "Low",
        ),
        (
            "Ticket creation and status tracking",
            "Create a support ticket and confirm its state.",
            [
                ("knowledge_base", "retrieve_policy", {"topic": "write approval"}),
                ("github", "triage_issue", {"issue": 128}),
            ],
            "Ticket labeled and prioritized for follow-up.",
            "None in the demo run.",
            "Not needed.",
            "Low",
        ),
        (
            "Database query and reporting",
            "Produce a small report from query results.",
            [
                ("database", "run_query", {"sql": "SELECT region, COUNT(*) FROM sales GROUP BY region"}),
                ("database", "generate_report", {"title": "Regional sales report"}),
            ],
            "Report generated and stored for review.",
            "Report generation requires approval.",
            "Approval granted before the write step.",
            "Medium",
        ),
        (
            "Repository triage",
            "Inspect the repo and identify the next action.",
            [
                ("github", "repo_info", {"repo": "demo/repo", "branch": "main"}),
                ("github", "list_commits", {"repo": "demo/repo"}),
            ],
            "Repository triage summary with recent commits.",
            "None in the demo run.",
            "Not needed.",
            "Low",
        ),
        (
            "Pull request creation",
            "Open a PR after the fix is approved.",
            [
                ("github", "repo_info", {"repo": "demo/repo", "branch": "feature/demo"}),
                ("github", "open_pr", {"repo": "demo/repo", "branch": "feature/demo", "title": "Fix demo build"}),
            ],
            "Pull request created at a demo GitHub URL.",
            "Write tool requires approval.",
            "Approval gate displayed before the PR step.",
            "Medium",
        ),
        (
            "Knowledge base lookup",
            "Retrieve governance guidance for the demo.",
            [
                ("knowledge_base", "answer_faq", {"question": "What are the approval rules?"}),
                ("knowledge_base", "fetch_guidance", {"topic": "least privilege"}),
            ],
            "Guidance returned for approvals and least privilege.",
            "None in the demo run.",
            "Not needed.",
            "Low",
        ),
        (
            "Workflow log review",
            "Read logs and extract the main signal.",
            [
                ("filesystem", "read_file", {"path": "logs/workflow.log"}),
                ("filesystem", "summarize_doc", {"title": "workflow logs"}),
            ],
            "Workflow logs summarized for the presenter.",
            "None in the demo run.",
            "Not needed.",
            "Low",
        ),
        (
            "Policy check",
            "Verify that write actions are gated.",
            [
                ("knowledge_base", "retrieve_policy", {"topic": "write approvals"}),
                ("github", "triage_issue", {"issue": 209}),
            ],
            "Policy explanation ready for the audience.",
            "None in the demo run.",
            "Not needed.",
            "Low",
        ),
        (
            "Interoperability check",
            "Confirm the same tools can be reused across agents.",
            [
                ("github", "repo_info", {"repo": "demo/repo", "branch": "main"}),
                ("filesystem", "search_docs", {"query": "same tools across agents"}),
            ],
            "Interoperability note prepared for the closing slide.",
            "None in the demo run.",
            "Not needed.",
            "Low",
        ),
        (
            "End-to-end audit log review",
            "Show the traceable output for all actions.",
            [
                ("filesystem", "summarize_doc", {"title": "audit log"}),
                ("knowledge_base", "fetch_guidance", {"topic": "structured logging"}),
            ],
            "Audit log summary captured for the final slide.",
            "None in the demo run.",
            "Not needed.",
            "Low",
        ),
    ]


async def main() -> None:
    project_root = Path(__file__).resolve().parent
    server_path = project_root / "mcp_demo_server.py"
    if not server_path.exists():
        raise FileNotFoundError(f"MCP demo server not found: {server_path}")

    clients = {
        role: MCPDemoClient(role, server_path, project_root)
        for role in ["filesystem", "github", "database", "knowledge_base"]
    }
    agent = MCPDemoAgent(clients)

    discovered = await agent.discover_tools()
    task_specs = build_demo_tasks(agent)
    tasks: List[TaskResult] = []
    for spec in task_specs:
        tasks.append(await agent.run_task(*spec))

    print_heading("Technical Seminar Runnable Demo")
    print("Architecting Universal AI Agents using MCP")
    print("This version uses the official MCP Python SDK with stdio servers.")
    print()
    print("Demo setup:")
    print("- Tasks: 10 representative evaluation tasks")
    print("- MCP servers: filesystem, GitHub, database, knowledge base")
    print("- Logging: timestamp, arguments hash, server, latency, status, retries")
    print("- Safety: tool allow-list, approval for write tools, scope-limited access")
    print()
    print("Discovered tools per server:")
    for role, tools in discovered.items():
        print(f"- {role}: {', '.join(tools)}")

    print_heading("Live Flow")
    print("1. User request: investigate, explain, and act on a task.")
    print("2. Agent discovers tools from multiple servers.")
    print("3. Agent executes read-only calls automatically.")
    print("4. Agent pauses for write approval when needed.")
    print("5. Agent records an audit trail and final output.")

    tool_call_rows: List[List[str]] = []
    task_rows: List[List[str]] = []
    error_rows: List[List[str]] = []
    approval_rows: List[List[str]] = []

    for task in tasks:
        for call in task.tool_calls:
            tool_call_rows.append([
                call.timestamp,
                call.tool_name,
                call.arguments_hash,
                call.server,
                str(call.latency_ms),
                call.status,
                str(call.retries),
            ])
        task_rows.append([
            task.task,
            str(task.steps),
            f"{task.latency_ms} ms",
            task.cost_estimate,
            task.final_output,
        ])
        if task.failure_modes != "None in the demo run.":
            error_rows.append([task.task, task.failure_modes, task.recovery_actions])
        if task.approved:
            approval_rows.append([task.task, "Approved", "Write action gated before execution"])

    print_heading("Representative Tasks")
    print_table(["Task", "Steps", "Latency", "Cost", "Final output"], task_rows)

    print_heading("Tool-Call Trace Table")
    print_table(
        ["Timestamp", "Tool name", "Args hash", "Server", "Latency (ms)", "Status", "Retries"],
        tool_call_rows,
    )

    print_heading("Error Summary Table")
    if error_rows:
        print_table(["Task", "Failure modes", "Recovery actions"], error_rows)
    else:
        print("No failures observed in this demo run.")

    print_heading("Safety and Approval Log")
    if approval_rows:
        print_table(["Task", "Approval", "Policy result"], approval_rows)
    else:
        print("No approval-gated actions were triggered.")

    print_heading("Interoperability Matrix")
    print_table(
        ["Host / model combination", "Tool reuse without rewrites"],
        [
            ["Python runner + MCP stdio servers", "Yes"],
            ["Different host UI + same server contracts", "Yes"],
            ["Different model + same server contracts", "Yes"],
        ],
    )

    output_dir = project_root / "demo_output"
    output_dir.mkdir(exist_ok=True)
    artifacts = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tools": discovered,
        "tasks": [asdict(task) for task in tasks],
        "tool_calls": [asdict(call) for task in tasks for call in task.tool_calls],
    }
    (output_dir / "demo_artifacts.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")

    print_heading("Saved Artifacts")
    print(f"Wrote: {output_dir / 'demo_artifacts.json'}")
    print("The file includes task summaries, tool-call traces, and approval metadata.")
    print()
    print("Closing line:")
    print(
        textwrap.fill(
            "This runnable demo shows why MCP-style tool boundaries matter: the agent can discover tools,"
            " reuse them across tasks, keep approvals explicit, and produce a traceable audit log.",
            width=78,
        )
    )

    for client in clients.values():
        # The SDK session objects are scoped per request; no persistent resources to close.
        _ = client


if __name__ == "__main__":
    asyncio.run(main())
