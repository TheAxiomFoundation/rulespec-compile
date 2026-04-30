# Validation And Oracles

`rulespec-compile` now has two validation layers:

- the compiler harness
- the RuleSpec vs PolicyEngine validation pipeline

They are related, but they solve different problems.

## 1. Compiler harness

Run the built-in scorecard:

```bash
rulespec-compile harness
rulespec-compile harness --json
```

The harness is the fast objective loop for compiler work. It checks:

- generic compile features
- graph/import/export behavior
- lowered bundle consistency
- generated runtime behavior
- batch execution
- shipped example oracles

Opt into curated live-stack checks against sibling repos and
artifacts such as `rules-us`, `rules-us-co`, and `axiom-encode`:

```bash
rulespec-compile harness --include-live
```

Focused runs:

```bash
rulespec-compile harness --case branching_formula
rulespec-compile harness --case branching_batch_execution
rulespec-compile harness --case oracle_snap_example
```

External oracle cases are opt-in:

```bash
rulespec-compile harness --include-external
```

That currently enables PolicyEngine-backed checks when `policyengine-us` is
installed.

## 2. Validation pipeline

Run per-household sample validation:

```bash
rulespec-validate --mode sample
```

Run the full CPS/vectorized lane:

```bash
rulespec-validate --mode full
```

## Current execution modes

Reports now label which RuleSpec execution path produced the result:

- `compiled_example`: compiled `.yaml` example calculators on one household at a time
- `compiled_batch`: lowered-program batch execution over a full DataFrame
- `policyengine_household`: one-household PolicyEngine evaluation
- `policyengine_microsim`: PolicyEngine microsimulation over CPS data

## What is being compared

Today, the shipped policy examples are:

- `/Users/maxghenis/TheAxiomFoundation/rulespec-compile/examples/eitc.yaml`
- `/Users/maxghenis/TheAxiomFoundation/rulespec-compile/examples/ctc.yaml`
- `/Users/maxghenis/TheAxiomFoundation/rulespec-compile/examples/snap.yaml`

The harness compares compiled outputs from those files against Python reference
calculators. The external harness lane can also compare compiled SNAP output to a
PolicyEngine household oracle.

The validation pipeline compares RuleSpec-side results to PolicyEngine at a larger
scale, including the full CPS/microsim path.

## When to use which tool

- Use `rulespec-compile harness` while changing the compiler itself.
- Use `rulespec-validate --mode sample` for quick behavior checks against PolicyEngine.
- Use `rulespec-validate --mode full` for broader empirical comparison on CPS data.

## Current boundaries

- External PolicyEngine harness cases are optional and depend on local install.
- The batch executor supports the current validated lowered subset, including
  limited `if` / `elif` / `else` blocks.
- Unsupported lowered constructs still fail loudly rather than degrading into
  approximate validation.

## Practical workflow

Typical compiler iteration loop:

1. Add or update a harness case.
2. Make the compiler pass it.
3. Run `rulespec-compile harness`.
4. Run `rulespec-validate --mode sample` if the change affects shipped examples.
5. Run `rulespec-validate --mode full` for bigger validation changes.
