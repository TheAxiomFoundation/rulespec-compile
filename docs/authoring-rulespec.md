# Authoring RuleSpec `.yaml`

This guide is for writing RuleSpec source files that compile cleanly through the
current `rulespec-compile` pipeline.

## File Shape

A current RuleSpec file uses the structured `rulespec/v1` envelope:

- `format: rulespec/v1`
- optional `module.summary` text for the source provision
- optional top-level `source:` metadata
- optional `imports:` list of local `.yaml` modules
- `rules:` list containing versioned parameter and derived rules

For canonical RuleSpec trees, use citation paths like
`statutes/26/32/c/2/A.yaml`, `regulations/...`, or `policies/...`. Avoid
generic entrypoint names like `main.yaml`.

The compiler uses canonical `statutes/...`, `regulations/...`, or
`policies/...` paths as module/rule identity in merged graphs, lowered bundles,
and generated citation metadata. For ad hoc files outside those roots, it falls
back to the file leaf name.

For real policy work, author RuleSpec as checked-in `.yaml` files with rule
citations and source metadata in the file itself. Do not treat RuleSpec as an ad
hoc embedded string format.

Current file example:

```yaml
format: rulespec/v1
module:
  summary: |-
    Example Tax Code imposes a 20 percent tax on taxable income.
source:
  citation: Example Tax Code
  accessed: '2026-01-01'
rules:
  - name: income_tax_rate
    kind: parameter
    dtype: Rate
    source: Example Tax Code
    source_url: https://example.test/tax
    versions:
      - effective_from: '2026-01-01'
        formula: '0.2'
  - name: income_tax
    kind: derived
    entity: TaxUnit
    period: Year
    dtype: Money
    unit: USD
    source: Example Tax Code
    source_url: https://example.test/tax
    versions:
      - effective_from: '2026-01-01'
        formula: taxable_income * income_tax_rate
```

Current jurisdiction examples live in:

- `/Users/maxghenis/TheAxiomFoundation/rules-us/statutes/26/3101/a.yaml`
- `/Users/maxghenis/TheAxiomFoundation/rules-us/statutes/26/45A/a.yaml`
- `/Users/maxghenis/TheAxiomFoundation/rules-us/statutes/26/63/c.yaml`
- `/Users/maxghenis/TheAxiomFoundation/rules-us/statutes/26/63/c/5.yaml`

## Parameter Rules

Parameter rules hold versioned numeric values used by derived formulas.

```yaml
- name: oasdi_wage_tax_rate
  kind: parameter
  dtype: Rate
  source: 26 USC 3101(a)
  source_url: https://www.law.cornell.edu/uscode/text/26/3101
  versions:
    - effective_from: '1990-01-01'
      formula: '0.062'
```

## Derived Rules

Derived rules are the compiled formulas. Entity-scoped rules usually include:

- `kind: derived`
- `entity`
- `period`
- `dtype`
- one or more `versions:` entries with formulas

Scalar derived rules can omit `entity` when they behave like helper
values or statute-level scalar concepts.

Real policy-style example:

```yaml
- name: oasdi_wage_tax
  kind: derived
  entity: TaxUnit
  dtype: Money
  period: Year
  unit: USD
  source: 26 USC 3101(a)
  source_url: https://www.law.cornell.edu/uscode/text/26/3101
  versions:
    - effective_from: '1990-01-01'
      formula: wages * oasdi_wage_tax_rate
```

## Supported formula subset

The current compiler supports:

- assignments plus a final `return`
- arithmetic
- comparisons
- boolean operators
- ternaries like `cond ? a : b`
- indexed lookup access like `threshold[n]`
- `abs`, `ceil`, `floor`, `max`, `min`, `round`
- limited `if` / `elif` / `else` blocks when every reachable path returns

Example with statement-block branching:

```yaml
- name: income_tax
  kind: derived
  entity: TaxUnit
  period: Year
  dtype: Money
  source: Example Tax Code
  source_url: https://example.test/tax
  versions:
    - effective_from: '2026-01-01'
      formula: |-
        if is_joint:
          rate = 0.1
        else:
          rate = 0.2
        return taxable_income * rate
```

The compiler fails loudly on unsupported constructs instead of guessing.

## Source connection

The source connection is part of the model, not optional decoration.

- top-level `source:` describes the file-level authority or archive context
- rule-level `source:` fields tie formulas and values back to statutes,
  regulations, or guidance
- rule-level `source_url:` fields preserve the public URL used by reviewers

If you are authoring real policy, prefer real `.yaml` modules on disk over inline
snippets in host-language strings.

## Imports

Current `rulespec/v1` imports are local file paths:

```yaml
imports:
  - ./shared.yaml
```

Imports can also resolve through configured module roots or package aliases:

- `rulespec.toml` `[module_resolution].roots`
- `rulespec.toml` `[module_resolution.packages]`
- CLI `--module-root`
- CLI `--package`

## Temporal definitions

Both parameter and derived rules can have multiple dated entries:

```yaml
- name: income_tax
  kind: derived
  entity: TaxUnit
  period: Year
  dtype: Money
  source: Example Tax Code
  source_url: https://example.test/tax
  versions:
    - effective_from: '2025-01-01'
      formula: taxable_income * 0.1
    - effective_from: '2026-01-01'
      formula: taxable_income * 0.2
```

Compile these with `--effective-date YYYY-MM-DD`.

## Good current patterns

- Keep rule names specific enough to be meaningful in generated outputs.
- Put `source:` and `source_url:` on every rule in jurisdiction repos.
- Keep formulas straight-line when possible.
- Use ternaries for simple expression-level branching.
- Use statement `if` blocks only when branch-local assignments make them clearer.
- Use `--select-output` to compile the smallest reachable subgraph.

## Current boundaries

The current generic compiler still does not support:

- loops
- `match` / `case`
- `try` / `except`
- arbitrary helper calls
- generic attribute access
- wildcard re-exports
- nested namespace chains beyond `alias.value`
- remote imports or package registries

For the exact supported scope, see the main README.
