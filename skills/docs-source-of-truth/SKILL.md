---
name: docs-source-of-truth
description: Maintain project documentation as the source of truth for implementation details. Use when answering questions about this repository, changing code/configuration/CLI/data contracts, or when documentation in docx/ or docs/ may be incomplete or stale and must be consulted or updated.
---

# Documentation Source of Truth

Keep the repository's Markdown documentation synchronized with its actual implementation. Consult documentation before making changes and update every affected document in the same task.

## Documentation roots

Resolve the documentation directory before working:

- Use `docx/` when that directory exists; it is the preferred project documentation root requested by the user.
- Otherwise use the existing `docs/` directory. In the current repository, documentation is in `docs/`.
- Also inspect `README.md` when the task concerns installation, common commands, project scope, or user-facing entry points.

Do not silently treat generated reports, notebooks, logs, or source comments as authoritative documentation. Use them as evidence only, unless the user explicitly asks to document them.

## Required workflow

Follow this workflow for every repository task, including tasks that initially look like simple code edits.

1. Inspect the repository structure, `git status`, the relevant source/configuration files, and the documentation root.
2. Search the documentation root and `README.md` for the component, symbol, command, path, configuration key, data format, or workflow involved. Use `rg` and read the relevant Markdown files before forming an implementation plan.
3. Treat the documentation as the project’s current contract, but verify it against the implementation. Identify contradictions, omissions, obsolete commands, and undocumented side effects.
4. Make the requested implementation change while preserving unrelated user changes.
5. Immediately identify every Markdown file affected by the change. Update all affected sections in the same task; do not leave a known stale document for a later pass.
6. Rewrite affected documentation to describe the actual current version, not a diff or a promise. Include concrete paths, commands, parameters, defaults, inputs/outputs, dependencies, invariants, failure modes, and operational limitations when relevant.
7. Validate the implementation and documentation proportionally to the change. Prefer existing tests, lint/type checks, dry-runs, configuration validation, and command help output. If a check cannot be run, state why and do not claim it passed.
8. Review the final diff for synchronization: code, configuration, CLI examples, file paths, and documented behavior must agree.

## Selecting and updating documents

Use the narrowest existing document that owns the changed behavior, then update cross-references where needed. Typical ownership in this repository is:

- architecture, stage ordering, artifacts, and contracts → `ARCHITECTURE.md`;
- datasets, schemas, preparation, and sources → `DATA.md`;
- model families and model configuration → `MODELS.md`;
- pretraining and PEFT → `PRETRAIN.md`;
- RL/alignment → `RL.md`;
- benchmarks and metrics → `BENCHMARKS.md`;
- notebooks → `NOTEBOOKS.md`;
- Airflow orchestration → `AIRFLOW.md`;
- MLflow and experiment tracking → `MLFLOW.md`.

This mapping is a starting point, not a substitute for reading the files. If a change spans multiple areas, update each owning document and the relevant navigation or overview links.

If no suitable document exists, create a focused Markdown file under the resolved documentation root and link it from the nearest overview document or `README.md`. Do not create duplicate documents for the same component.

## Documentation quality rules

- Write in the language already used by the relevant document unless the user requests another language.
- Prefer a self-contained explanation over vague references to source code.
- Keep examples executable and consistent with current paths, option names, environment variables, and defaults.
- Explain both the normal path and important edge cases: fallback behavior, required credentials, network/model requirements, resource usage, and error conditions.
- Preserve useful existing detail, but remove or correct statements that are no longer true.
- Mark assumptions and unverified behavior explicitly. Never invent results, APIs, metrics, or file contents.
- Keep generated artifacts out of conceptual documentation unless they are part of the supported contract.
- A documentation-only change must still be checked for stale links, commands, paths, and references to renamed symbols.

## Completion checklist

Before reporting completion, confirm:

- the relevant documentation was consulted before the change;
- every affected Markdown file was updated or explicitly confirmed unaffected;
- the documentation describes the complete current behavior and not merely the edit;
- examples and paths were checked against the repository;
- tests or other validation were run when practical;
- any unresolved discrepancy or skipped validation is reported clearly.
