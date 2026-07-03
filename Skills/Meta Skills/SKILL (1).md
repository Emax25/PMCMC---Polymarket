---
name: session-retrospective
description: |
  Analyze the current Claude Code session and generate a retrospective summary of lessons learned.
  Use when the user wants to: (1) reflect on what was accomplished in a session, (2) summarize
  lessons learned for future reference, (3) create a blog-post-friendly writeup of a coding session,
  (4) understand what went wrong and how it was fixed, or (5) document techniques discovered.
  Triggers: "retrospective", "what did we learn", "session summary", "lessons learned", "retro".
---

# Session Retrospective

Generate a reflective summary of the current session, suitable for future reference or blog post sharing.

## Workflow

1. **Retrieve session**: Run `scripts/get-session.sh` to get the current session JSONL
2. **Parse and analyze**: Extract user messages, assistant responses, tool uses, and corrections
3. **Identify key moments**: Problems encountered, solutions found, mistakes made, techniques learned
4. **Generate retrospective**: Output markdown to console

## Running the Script

```bash
# Current session (uses CLAUDE_SESSION_ID env var)
~/.claude/skills/session-retrospective/scripts/get-session.sh

# Specific session
~/.claude/skills/session-retrospective/scripts/get-session.sh <session-id>
```

The script outputs raw JSONL. Parse it to extract:
- `"type":"user"` entries for user messages
- `"type":"assistant"` entries for Claude responses
- Look for `tool_result` with `is_error: true` for rejected/failed actions
- Look for user corrections following assistant messages

## Analysis Focus Areas

When analyzing the session, identify:

| Area | What to Look For |
|------|------------------|
| **Problems & Solutions** | Errors encountered, debugging steps, what finally worked |
| **Key Decisions** | Why certain approaches were chosen, trade-offs considered |
| **Techniques Discovered** | New tools, commands, patterns, or methods learned |
| **Mistakes & Corrections** | User rejections, wrong assumptions, course corrections |

## Output Format

Generate markdown structured for potential blog sharing:

```markdown
# Session Retrospective: [Brief Title]

**Date**: [date]
**Duration**: [if determinable from timestamps]
**Project**: [working directory or project name]

## TL;DR

[2-3 sentence summary of what was accomplished and the key takeaway]

## What We Set Out To Do

[Brief description of the initial goal/problem]

## The Journey

### [Challenge/Phase 1 Title]

[What happened, what was tried, what worked/didn't work]

**Key insight**: [The important lesson from this phase]

### [Challenge/Phase 2 Title]

[Continue for each significant phase...]

## Mistakes Made (And What I Learned)

- **[Mistake 1]**: [What went wrong] → [What the fix/lesson was]
- **[Mistake 2]**: ...

## Techniques Worth Remembering

- **[Technique 1]**: [Brief description of the technique and when to use it]
- **[Technique 2]**: ...

## Key Takeaways

1. [Most important lesson]
2. [Second most important lesson]
3. [Third most important lesson]

---

*Generated from Claude Code session retrospective*
```

## Writing Guidelines

- **Be specific**: Include actual error messages, command names, file paths where relevant
- **Show the process**: Document the messy middle, not just the clean solution
- **Extract transferable lessons**: Frame insights so they apply beyond this specific session
- **Honest about mistakes**: The most valuable lessons often come from what went wrong
- **Conversational tone**: Write as if explaining to a fellow developer over coffee
