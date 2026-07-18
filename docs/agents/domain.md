# Domain docs

This is a single-context repository. Engineering skills should use the
repository's domain documentation as follows.

## Before exploring

- Read `CONTEXT.md` at the repository root when it exists.
- Read architectural decision records under `docs/adr/` that touch the area
  being changed.
- If either location does not exist, proceed without flagging its absence.

## Vocabulary

Use domain terms as defined in `CONTEXT.md` in issues, refactor proposals,
hypotheses, and test names. If a needed concept is absent, reconsider whether
the project uses another name or update the glossary when the concept becomes
part of an accepted design.

## Architectural decisions

If proposed work contradicts an existing record under `docs/adr/`, identify
the conflict explicitly instead of silently overriding the decision.
