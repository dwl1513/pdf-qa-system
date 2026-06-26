# TeamOS Long-Running Loop Prompt (Phase 1)

Drive this with Claude Code's `/loop`:

```
/loop 10m <paste the "Loop body" section below>
```

This is the **Phase 1** driver: a human-launched `teamos run` session (CC inside tmux) that pulls
the TeamOS task queue and works it down, while a human can `teamos attach` to watch or intervene.
**Phase 2** will replace this with a TeamOS-supervised `LocalSpawnRuntime` driving headless
`claude -p` for unattended runs — not implemented yet.

The capability token injected by `teamos run` (`TEAMOS_TOKEN`) auto-refreshes inside the MCP
`GatewayClient` at ~10% remaining lifetime, so an 8h run survives the 4h TTL with no manual step.

---

## First turn only

Call `teamos_triage` with `{ session_id: $TEAMOS_SESSION_ID, user_id: <you>, intent: "..." }`.
Triage detects the workspace from `CLAUDE_PROJECT_DIR`, binds it to this session, and reports the
repo/branch/host. Do this exactly once.

## Loop body (repeats every interval)

1. **Check the budget.** `teamos_run_status { session_id: $TEAMOS_SESSION_ID }`.
   If `should_stop` is true → announce the `reasons` and STOP the loop (do not pull more tasks).
2. **Claim work.** `teamos_task_next { session_id: $TEAMOS_SESSION_ID }`.
   If `{ task: null }` → the queue is empty; STOP the loop.
3. **Do the task** in the bound workspace. All code/doc edits and **all git commits are done by
   YOU in the workspace** — the gateway never touches files.
4. **Report the outcome:**
   - Success → `teamos_task_update { id, status: "shipped" }`, then
     `teamos_run_signal { session_id: $TEAMOS_SESSION_ID, kind: "success" }`.
   - Blocked / needs a human decision → `teamos_ask_human { ... }`, then
     `teamos_run_signal { session_id: $TEAMOS_SESSION_ID, kind: "failure" }`.
5. Continue to the next iteration (the loop interval handles pacing; the token refreshes itself).
