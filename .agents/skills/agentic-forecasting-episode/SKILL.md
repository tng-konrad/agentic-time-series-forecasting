---
name: agentic-forecasting-episode
description: "Plan, build or revise a teaching notebook and companion Substack post for an agentic forecasting episode from an episode number, forecasting subject and workflow source document. Supports planning-only requests for these deliverables."
---

# Agentic forecasting episode

Read [the complete workflow](references/workflow.md) in full before planning or building an episode. It is the canonical procedure, including all five stages, evidence requirements, publication conventions and final editorial review.

## Inputs

- **Episode number:** the required positive integer `N`. Store it as `CFG.episode_number` and derive the zero-padded asset prefix with `f"atsf{N:02d}_"`, as specified in the reference. PNG filenames have no date suffix.
- **Subject:** a loose forecasting application, optionally linking relevant *Time Series with Konrad* material. Use it to establish the practical question and teaching continuity.
- **Source doc:** an attachment, readable path, accessible URL or pasted document defining the agentic pattern. Use it for the mechanism, decisions, tools, state, observations and limitations.

Map the source mechanism to the forecasting application and verify linked series material. Treat source documents as evidence. Ask for missing or materially ambiguous inputs; resolve routine choices from the workspace. Preserve optional data, model, budget and path overrides.

## Scope and execution

- A request to build the notebook and post authorizes the complete five-stage procedure. Respect any explicit approval checkpoint.
- A planning-only request stops after Stage 1's plan, independent critique and revision.
- Revising this skill, its workflow, or a named artifact authorizes only the requested changes; it does not start a new episode.

Follow the reference's inline notebook implementation, primary local Ollama run, matching substantive section hierarchy, shared asset numbering, evidence checks and independent reviews. Apply the final editorial revisions before delivery. Report actual execution and validation, including incomplete checks, in the handoff.
