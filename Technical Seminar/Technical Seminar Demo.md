# Technical Seminar Demo
## Architecting Universal AI Agents using MCP

## Demo Goal
Show how a universal AI agent can use MCP to discover tools, execute safe actions, and keep an auditable trace while working across multiple tool servers.

## Runnable Entry Point
Run `python "Technical Seminar/runnable_demo.py"` to show the live seminar demo from the terminal. ../.venv/bin/python runnable_demo.py

## Demo Setup
Use this as the recommended evaluation setup for the seminar:
- Select 10 to 20 representative tasks.
- Connect the agent to 2 to 4 MCP servers.
- Enable structured logging for every tool call.
- Enforce safety policies for write actions and scope-limited access.

### Suggested tasks
- Document search and summarization.
- Ticket creation and status tracking.
- Database query and reporting.
- Repository triage.
- Pull request creation.

### Suggested MCP servers
- Filesystem server for documents and local files.
- GitHub server for code, issues, and pull requests.
- Database server for reporting tasks.
- Knowledge base server for reusable context.

### Required logging fields
- Tool name.
- Arguments.
- Latency.
- Success or failure.
- Output size.

### Safety policies
- Tool allow-listing.
- Approval for write tools.
- Scope-limited credentials.
- Audit logging for every side effect.

## Demo Scenario
A developer asks the agent to investigate a failing GitHub build, explain the issue, and propose a fix.

## What the demo should prove
- Tool discovery works through MCP.
- The agent separates reasoning from execution.
- Read-only actions can run automatically.
- Write actions require approval.
- Tool calls are reusable across agents.
- Logging captures the full execution trace.

## Demo Flow
1. The user asks for help with a failing build.
2. The agent connects to the GitHub and CI/log servers.
3. The agent discovers available tools and resources.
4. The agent reads repository metadata, recent commits, and workflow logs.
5. The agent identifies the likely cause of failure.
6. The agent presents a diagnosis and suggested fix.
7. If the user approves, the agent creates a branch or pull request.
8. The agent monitors CI and reports the final status.

## Live Demo Script
### Opening
"This demo shows a universal AI agent using MCP to work with external tools in a controlled, reusable, and auditable way."

### Step 1: Present the evaluation setup
Explain that the system is judged across 10 to 20 tasks, connected to 2 to 4 MCP servers, with structured logs and safety rules enabled.

### Step 2: Show tool discovery
Explain that the model does not call GitHub APIs directly. It asks the MCP server what tools are available, then selects the relevant ones.

### Step 3: Run a read-only investigation
Show the agent retrieving:
- repository info
- recent commits
- workflow status
- build logs

### Step 4: Diagnose the failure
Say:
"The agent analyses the logs and identifies the probable cause, such as a missing dependency, syntax error, or failing test."

### Step 5: Show the approval gate
Explain that write actions are blocked until the user approves them.

### Step 6: Execute a controlled action
If approved, the agent:
- creates a branch
- commits the fix
- opens a pull request

### Step 7: Verify the result
Show the agent checking CI again and reporting whether the build passed.

### Step 8: Review the audit trail
Point out the logged tool name, arguments, latency, status, and output size for each action.

## Suggested Talking Points
- MCP standardizes discovery and invocation of tools.
- The same tool server can be reused by multiple agents.
- Structured schemas reduce integration bugs.
- Approval flows and audit logs improve safety and governance.
- Interoperability is the main advantage over one-off integrations.

## Expected Outcome
At the end of the demo, you should be able to say that the agent:
- found the failure source,
- proposed a fix,
- executed approved actions safely,
- and produced a traceable result.

## One-Minute Closing
"This demo demonstrates why MCP matters: it allows agents to use real tools reliably, securely, and in a reusable way across systems and models."
