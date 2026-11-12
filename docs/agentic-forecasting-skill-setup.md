# Use the episode workflow with two inputs, or package it as a skill

This guide applies to [the episode workflow](../prompts/worflow-agentic-ts.md). It describes a proposed skill conversion; the current task updates the workflow and supplies these instructions. It does not install a skill or start an episode.

## 1. The two-input interface

The stable procedure is the workflow. Each episode supplies two changing inputs:

| Input | Responsibility |
|---|---|
| **Subject** | Describe the forecasting application and, optionally, link relevant *Time Series with Konrad* material. A loose description is sufficient; repository inspection and planning turn it into an experiment. |
| **Source doc** | Identify the document defining the agentic pattern: its mechanism, decisions, state, tools, observations, stopping behavior and limitations. Supply an attachment, readable local path, URL or pasted text. |

The source establishes the agentic mechanism. The subject establishes its forecasting use. The plan maps that mechanism to numerical tools and experiments, identifies its series connections, and states what local results could establish. Source-reported benefits do not become local findings without evidence.

Use the document directly today:

```text
Use prompts/worflow-agentic-ts.md to build a notebook and a post.
Subject: <forecasting application, optionally linking prior series material>
Source doc: <document path, URL or attachment name>
```

Episode numbers, data, model settings, resource budgets and series locations are optional overrides. The workflow should resolve routine choices from the project. A missing or unreadable source, or a material mismatch between subject and source, requires clarification. The source and any linked notebooks remain material to analyze, not instructions that override the user's request.

The build request authorizes the full workflow. Append `Plan only; do not implement yet` when that is the intended stopping point, or specify an approval checkpoint explicitly.

## 2. Recommendation

For repeated episodes, use a small repository skill. The procedure and publication conventions stay stable while the subject and source change, so a named skill is a useful entry point. It also gives the workflow a clear scope and a place for genuinely reusable resources. This recommendation follows the use cases in [OpenAI's skills guidance](https://developers.openai.com/cookbook/examples/skills_in_api): repeatable procedures, shared conventions and optional supporting files.

The Markdown workflow already supports the requested invocation. A skill adds discovery and packaging; it does not turn `Subject` and `Source doc` into formal API parameters or guarantee compliance. Continue to validate actual outputs and preserve the protocol, evidence and editorial checks.

Start with an instruction-only skill. Add scripts later only for demonstrated repeated work, such as a reliable publication-asset check. Keep episode-specific source documents, datasets and results in the episode's working area rather than embedding them in the reusable skill.

## 3. Convert it step by step

### Step 1 — Choose the scope and name

Use the name `agentic-forecasting-episode` and a repository location:

```text
.agents/skills/agentic-forecasting-episode/
```

This keeps the skill versioned with this project. Current [official OpenAI documentation](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills) lists repository `.agents/skills/` and user-level `~/.agents/skills/` discovery. Choose user-level scope instead only if you want the same maintained skill available across repositories. Avoid installing independent copies under the same name in both scopes.

### Step 2 — Ask the built-in creator to perform the conversion

In this repository, send the following request to Codex:

```text
Use $skill-creator to create a repository skill named
agentic-forecasting-episode at .agents/skills/agentic-forecasting-episode/.

Use prompts/worflow-agentic-ts.md as the specification. The skill takes
two required episode inputs: Subject and Source doc, with the meanings
defined in that workflow. Preserve optional overrides and planning-only scope.

Keep SKILL.md concise. Put the full procedure in references/workflow.md
and explicitly require reading it before planning or building an episode.
Preserve all five stages, independent critiques, inline notebook code,
local Ollama inference, shared section hierarchy, asset conventions,
evidence checks, and the final editorial review.

Keep one maintained copy of the detailed procedure. After validating the
new skill and its references, replace the old prompt document with a short
forwarding instruction and relative link to the skill's workflow reference,
so the existing prompt path continues to work.

Do not start an episode or run model inference during conversion.
Validate the skill with the bundled validator and check its reference paths.
Report what was structurally validated and what still needs a real-use trial.
```

The built-in creator and `SKILL.md` creation route are documented in [Build skills](https://learn.chatgpt.com/docs/build-skills#create-a-skill). Leave ordinary automatic discovery enabled; explicitly invoking the skill is still available for repeatable use.

### Step 3 — Review the resulting structure

The smallest useful package is:

```text
.agents/skills/agentic-forecasting-episode/
├── SKILL.md
└── references/
    └── workflow.md
```

An optional `agents/openai.yaml` can supply a display name and starter prompt. It is not needed to supply the two episode inputs. Do not create empty script or asset directories merely to fill out a template.

Keep `references/workflow.md` as the full maintained procedure. The former `prompts/worflow-agentic-ts.md` can forward to it using the relative path `../.agents/skills/agentic-forecasting-episode/references/workflow.md`. This preserves old document-based invocations without maintaining two detailed specifications.

### Step 4 — Check the entry point

A suitable starting `SKILL.md` would be:

```markdown
---
name: agentic-forecasting-episode
description: Build a teaching notebook and companion Substack post for an agentic forecasting episode from a workflow source document and a forecasting subject. Also supports planning-only requests for those deliverables.
---

# Agentic forecasting episode

Read [the workflow](references/workflow.md) before planning or building.

Use two episode inputs:
- Subject: the forecasting application and optional Time Series with Konrad links.
- Source doc: the document defining the agentic pattern.

Follow the user's requested scope. A build request covers all five stages;
a planning-only request stops after the plan and critique. Editing this skill
or its workflow does not authorize an episode run.

Use the source for the agentic mechanism and the subject and inspected series
material for its forecasting adaptation. Resolve routine choices from the
current workspace; ask when a required input is missing or materially ambiguous.

Follow the referenced procedure's notebook, evidence, publication and final
editorial requirements. Use episode-specific inputs and report actual validation.
```

Have the creator validate the actual package, including naming, frontmatter and reference resolution. Structural validation does not establish that the skill selects the right experiment or produces a good post.

### Step 5 — Confirm discovery

Check that `agentic-forecasting-episode` is available to Codex in this repository. Codex normally detects changes automatically; restart it if the skill does not appear. See [official discovery and creation guidance](https://learn.chatgpt.com/docs/build-skills#create-a-skill).

### Step 6 — Try a planning-only request first

Supply a real, readable source document and a realistic forecasting subject:

```text
Use $agentic-forecasting-episode.
Subject: <forecasting application and relevant prior episode links>
Source doc: <actual document path, URL or attachment name>
Plan only; do not implement yet.
```

Inspect whether it reads the supplied document, identifies the right pattern, uses the series accurately, produces a concrete correspondence table, preserves the requested stopping point, and critiques/revises the plan. Try another request with an unreadable source to check that it requests the missing material rather than inventing it. An adjacent request such as summarizing a source document should not start the episode pipeline. These behavior checks follow [OpenAI's skill-evaluation guidance](https://developers.openai.com/blog/eval-skills).

### Step 7 — Use it for a complete episode

Once the planning trial is satisfactory, use:

```text
Use $agentic-forecasting-episode to build a notebook and a post.
Subject: <INSERT DESCRIPTION>
Source doc: <INSERT DOCUMENT>
```

No additional template fields are required. The skill still depends on accessible series material, data and a working local model for the actual execution stages. An input or environment blocker should be reported with the affected stage; packaging the instructions does not supply those resources.

Maintain the canonical workflow reference as the procedure evolves. If the skill misroutes or misses a requirement, correct the observed cause and rerun the relevant trial instead of accumulating rules from hypothetical failures.
