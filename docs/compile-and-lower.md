# Compile And Lower

This guide covers the main `rulespec-compile` workflows.

## Compile to JavaScript, Python, or Rust

JavaScript:

```bash
rulespec-compile compile examples/simple_tax.yaml -o simple_tax.js
```

Python:

```bash
rulespec-compile compile examples/snap.yaml --python -o snap.py
```

Rust:

```bash
rulespec-compile compile examples/snap.yaml --rust -o snap.rs
```

The generated calculators all come from the same lowered program bundle.

## Compile a file graph

Entry files can import other `.yaml` files:

```bash
rulespec-compile compile examples/working_families/benefit_amount.yaml --python -o benefit_amount.py
```

Use canonical RuleSpec paths for real source trees rather than generic entrypoint
names. In other words, prefer `statute/26/32/c/2/A.yaml` over `main.yaml`.

The shipped file-graph example lives in:

- `/Users/maxghenis/TheAxiomFoundation/rulespec-compile/examples/working_families/benefit_amount.yaml`
- `/Users/maxghenis/TheAxiomFoundation/rulespec-compile/examples/working_families/base_amount.yaml`
- `/Users/maxghenis/TheAxiomFoundation/rulespec-compile/examples/working_families/phase_in_rate.yaml`
- `/Users/maxghenis/TheAxiomFoundation/rulespec-compile/examples/working_families/phase_in_cap.yaml`

That example exercises import aliases, selective imports, re-exports, exported
output aliases, and qualified external rule binding.

There is a short companion note with runnable commands in:

- `/Users/maxghenis/TheAxiomFoundation/rulespec-compile/examples/working_families/README.md`

You can also resolve bare imports through workspace roots:

```bash
rulespec-compile compile benefit_amount.yaml --python --module-root ./lib -o benefit_amount.py
```

Or package aliases:

```bash
rulespec-compile compile benefit_amount.yaml --python --package tax=./packages/tax -o benefit_amount.py
```

## Compile only selected public outputs

Use `--select-output` to prune the graph to the reachable subgraph for one output:

```bash
rulespec-compile compile examples/working_families/benefit_amount.yaml --binding phase_in_rate.rate=0.25 --select-output benefit_amount --python -o benefit_amount.py
```

This works against the public output surface, including exported aliases.

## Temporal compilation

If a file has more than one dated definition, pass an effective date:

```bash
rulespec-compile compile policy.yaml --effective-date 2025-01-01 --python -o policy.py
```

Without an effective date, the compiler errors instead of guessing.

## Bind external rules

One-off bindings:

```bash
rulespec-compile compile examples/working_families/base_amount.yaml --binding phase_in_rate.rate=0.25 --python -o base_amount.py
rulespec-compile compile examples/working_families/benefit_amount.yaml --binding phase_in_rate.rate=0.25 --select-output benefit_amount --python -o benefit_amount.py
```

Use `module_identity.symbol` when the bound rule lives in an imported file.
Bare names still work when they are unambiguous across the loaded graph.

JSON rule binding file:

```bash
rulespec-compile compile examples/working_families/benefit_amount.yaml --binding-file bindings.json --python -o benefit_amount.py
```

Real RuleSpec-side override artifact:

```bash
rulespec-compile compile examples/statute/26/32/b/2/A/base_amounts.yaml \
  --binding-file ../rules-us/irs/rev-proc-2023-34/eitc-2024.yaml \
  --effective-date 2024-06-01 \
  --python -o base_amounts.py
```

Repeated `--binding-file` flags merge in order, and inline `--binding` flags
override file values.

Current boundary: override artifacts are supported for scalar values and
integer-indexed tables. Non-integer keyed artifacts still fail loudly.

## Emit the lowered bundle

Lowering stops after graph resolution, temporal resolution, external rule binding, and
selected-output pruning:

```bash
rulespec-compile lower examples/snap.yaml -o snap.lowered.json
rulespec-compile lower examples/working_families/benefit_amount.yaml --binding phase_in_rate.rate=0.25 --select-output benefit_amount -o benefit_amount.lowered.json
```

The lowered JSON includes:

- public inputs
- resolved external values, each with source and `module_identity`
- ordered computations, each with `module_identity`
- typed outputs, each with `module_identity`

When an input comes from an imported rule, its public lowered/runtime name is
`module_identity.symbol` rather than the compiler's merged internal helper name.
Generated Python/JS calculators accept those qualified names directly, and the
Rust output provides `calculate_public(...)` for the same public-input contract.

This is the backend-neutral seam between RuleSpec source and target-specific codegen.
For canonical RuleSpec trees, `module_identity` comes from the `statute/...`,
`regulation/...`, or `legislation/...` path. For ad hoc files outside those
roots, the compiler falls back to the file leaf.

## Python API

Real file-backed program:

```python
from datetime import date
from pathlib import Path
from rulespec_compile import load_rulespec_program

program = load_rulespec_program(Path("examples/eitc.yaml"))

lowered = program.to_lowered_program(
    effective_date=date(2025, 1, 1),
    outputs=["eitc"],
)
python_code = lowered.to_python_generator().generate()
```

File graph:

```python
from datetime import date
from pathlib import Path
from rulespec_compile import load_rulespec_program

program = load_rulespec_program(Path("examples/working_families/benefit_amount.yaml"))
rust_code = program.to_rust_generator(
    effective_date=date(2025, 1, 1),
    outputs=["benefit_amount"],
).generate()
```

For real policy assets, prefer loading `.yaml` files from disk instead of
embedding RuleSpec in Python strings. That keeps the rule source, citations, and
module graph visible at the source level.

## What to use when

- Use `compile` when you want runnable JS, Python, or Rust.
- Use `lower` when you want the resolved backend-neutral artifact.
- Use `--select-output` when you want a condensed subtree rather than the full graph.
- Use the Python API when you are embedding compilation inside another tool.

## Current boundaries

The current compile/lower surface still rejects unsupported constructs loudly.
The main README is the source of truth for the exact validated subset.
