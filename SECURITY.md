# Security Guide

This document describes the current trust model, redaction boundary, MFA policy,
and operator re-authorization behavior for AI_Prototype.

## Trust Model

- Static skills in src/skills are admin-controlled code loaded from the local
  repository and executed through the main orchestrator process.
- Static skill execution therefore has the same OS-level permissions as the bot
  process and should be treated as trusted operational code.
- Dynamic synthesized skills are not executed inline in the orchestrator.
  They are validated (AST plus token checks) and then registered in an isolated
  dynamic-tool worker process.
- Telegram access is restricted to ADMIN_USER_ID, and sensitive tool requests
  can be moved into MFA or HITL intercepts before execution.

## Cloud Redaction Boundary

When the orchestrator routes to cloud reasoning, it uses a redaction-first
boundary before payloads leave the host.

Inside the boundary (redacted before cloud send):
- Message minimization (default): first and last system messages plus latest
  user message.
- Context and memory block sanitization (core working memory, archival memory,
  chat-history style blocks, and machine context markers).
- PII and environment-style line redaction patterns.
- Size truncation markers for privacy and payload control.

Outside the boundary (not cloud-redacted by this layer):
- Local System 1 processing and local tool execution.
- Local persistence layers (ledger, vector store, core memory JSON).
- Local logs and operator console/Telegram operational messages.
- Any route that explicitly opts into allow_sensitive_context=True.

## MFA Passphrase Requirements

MFA policy is enforced at startup and challenge time:

- MFA_PASSPHRASE must be configured before startup.
- Minimum passphrase length is 12 characters.
- Common authorization words are rejected, including: blue, password, admin,
  secret, aiden, and test.
- MFA challenge verification accepts either:
  - A valid TOTP code (when MFA_TOTP_SECRET is configured and pyotp is
    installed), or
  - A normalized whole-word match of the configured MFA passphrase.

## What Requires Re-Authorization

Can proceed without a new MFA challenge:
- Normal conversation handling and routing.
- Non-sensitive registered tool execution through SkillRegistry.
- Goal planning, synthesis candidate evaluation, and standard backlog updates
  that are not intercepted as privileged core updates.

Cannot proceed without MFA re-authorization:
- Privileged core-update requests routed through request_core_update.
- Any pending MFA challenge resumed from orchestrator state requires a valid
  passphrase/TOTP response before the deferred privileged action is resumed.

Operator note:
- MFA and HITL state is persisted, so a restart does not silently drop pending
  approvals.