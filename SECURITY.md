# Security and execution boundaries

PatchPilot is an experimental developer tool. Use disposable checkouts of
repositories you trust.

- The general CLI edits its selected directory in place. Its fixed test command
  executes code with the current user's host permissions; file-path checks do
  not sandbox that code.
- Docker test runs have resource limits and offline networking by default.
  Assertion probes use read-only mounts and root filesystems. These controls
  are not a claim of isolation against arbitrary hostile code or Docker escapes.
- Repository snippets, issues, and tool output are sent to the configured model
  provider. Confirm that you can share that code before running the Agent.
- Trajectories and error logs may include source code, personal paths, and
  sensitive output. They are ignored by Git. Review any excerpts before sharing.
- Store credentials in `.env` or process environment variables. `.env.example`
  contains placeholders only. Rotate any credential accidentally published.

## Reporting

For a non-sensitive bug, open an issue with versions and a minimal reproduction.
For a vulnerability, use the repository's private vulnerability reporting
channel when enabled. If it is unavailable, request a private contact channel
without publishing exploit details or credentials. No response-time guarantee
is offered for this research project.

The release scan checks common secret patterns and known local key values in
publishable files and reachable Git history. It is a heuristic check, not a
security audit or a guarantee that every sensitive value has been found.
