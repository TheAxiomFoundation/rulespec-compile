# Authoring `.yaml`

This guide is for writing RuleSpec source files that compile cleanly through the
current `rulespec-compile` pipeline.

## File shape

A `.yaml` file is made of:

- an optional top-level `source:` metadata block
- top-level rule definitions
- optional imports / exports

For canonical RuleSpec trees, use citation paths like `statute/26/32/c/2/A.yaml` or
`regulation/...`. Avoid generic entrypoint names like `main.yaml`.

The compiler uses canonical `statute/...`, `regulation/...`, or
`legislation/...` paths as module/rule identity in merged graphs, lowered
bundles, and generated citation metadata. For ad hoc files outside those roots,
it falls back to the file leaf name.

For real policy work, author RuleSpec as checked-in `.yaml` files with rule citations
and source metadata in the file itself. Do not treat RuleSpec as an ad hoc embedded
string format.

Shipped file example:

```rulespec
# /Users/maxghenis/TheAxiomFoundation/rulespec-compile/examples/simple_tax.yaml

source:
  citation: "Example Tax Code"
  accessed: 2025-01-01

rate:
  source: "example/tax/rate"

tax:
  entity: Person
  period: Year
  dtype: Money
  label: "Income Tax"
  from 2025-01-01:
    return income * 0.2
```

The more representative policy examples are:

- `/Users/maxghenis/TheAxiomFoundation/rulespec-compile/examples/eitc.yaml`
- `/Users/maxghenis/TheAxiomFoundation/rulespec-compile/examples/ctc.yaml`
- `/Users/maxghenis/TheAxiomFoundation/rulespec-compile/examples/snap.yaml`
- `/Users/maxghenis/TheAxiomFoundation/rulespec-compile/examples/working_families/benefit_amount.yaml`

## External Scalar And Table Rules

RuleSpec source files use one top-level rule shape. Some of those rules behave as
external scalar values or lookup tables rather than computed formulas.

Inline scalar external rule:

```rulespec
rate:
  source: "26 USC 1"
  values:
    0: 0.2
```

Indexed external rule table:

```rulespec
credit_pct:
  source: "26 USC 32(b)(1)"
  values:
    0: 7.65
    1: 34
    2: 40
    3: 45
```

Source-only external rule:

```rulespec
rate:
  source: "external/rate"
```

Source-only external rules must be bound explicitly at compile time with
`--binding` or `--binding-file`. For imported source-only rules, use
`module_identity.symbol`.

## Computed Rules

Computed rules are the compiled formulas. Entity-scoped rules usually include:

- `entity`
- `period`
- `dtype`
- one or more `from YYYY-MM-DD:` formula bodies

Scalar computed rules can omit `entity` when they behave like derived helper
values or statute-level scalar concepts.

Real policy-style example:

```rulespec
# excerpted from /Users/maxghenis/TheAxiomFoundation/rulespec-compile/examples/eitc.yaml

eitc:
  entity: TaxUnit
  period: Year
  dtype: Money
  label: "Earned Income Tax Credit"
  from 2025-01-01:
    n = min(n_children, 3)
    credit_pct_rate = credit_pct[n] / 100
    return round(max(0, earned_income * credit_pct_rate))
```

Scalar computed example:

```rulespec
snap_self_employment_cost_exclusion:
  label: "SNAP self-employment cost exclusion"
  description: "Reduction for production costs"
  from 2008-10-01:
    min(
      snap_nonfarm_self_employment_production_costs,
      snap_nonfarm_self_employment_gross_income,
    ) + snap_farm_self_employment_production_costs
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

```rulespec
tax:
  entity: Person
  period: Year
  dtype: Money
  from 2025-01-01:
    if is_joint:
      rate = 0.1
    else:
      rate = 0.2
    return wages * rate
```

The compiler fails loudly on unsupported constructs instead of guessing.

## Source connection

The source connection is part of the model, not optional decoration.

- top-level `source:` describes the file-level authority or archive context
- external-rule `source:` fields tie values back to statutes, regulations, or guidance
- variable labels and citations define the public rule surface

If you are authoring real policy, prefer real `.yaml` modules on disk over inline
snippets in host-language strings.

## Imports and exports

Local import:

```rulespec
import "./shared.yaml"
```

Aliased import:

```rulespec
import "./shared.yaml" as shared
```

Selective import:

```rulespec
from "./shared.yaml" import rate_public as rate, taxable_income
```

Export public names:

```rulespec
export tax, taxable_income
export taxable_income as base_income
```

Re-export from another module:

```rulespec
export from "./shared.yaml" import tax, rate as public_rate
```

## Bare imports

You can also import through configured module roots or package aliases:

```rulespec
from "tax/shared.yaml" import rate
```

Those are resolved through:

- `rulespec.toml` `[module_resolution].roots`
- `rulespec.toml` `[module_resolution.packages]`
- CLI `--module-root`
- CLI `--package`

## Temporal definitions

Both external rules and computed rules can have multiple dated entries:

```rulespec
tax:
  entity: Person
  period: Year
  dtype: Money
  from 2024-01-01:
    return wages * 0.1
  from 2025-01-01:
    return wages * 0.2
```

Compile these with `--effective-date YYYY-MM-DD`.

## Good current patterns

- Use explicit `values:` blocks for inline numeric external values.
- Keep formulas straight-line when possible.
- Use ternaries for simple expression-level branching.
- Use statement `if` blocks only when branch-local assignments make them clearer.
- Export only the public surface you want importers to depend on.
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
