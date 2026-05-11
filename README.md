# rulespec-compile

Compile RuleSpec `.yaml` files to standalone JavaScript, Python, and Rust calculators.

## Overview

`rulespec-compile` generates JS, Python, and Rust code from RuleSpec policy encodings. JavaScript output runs entirely in the browser with no server required. Python output can be imported and used in any Python application. Rust output is generated from the same lowered bundle for the current validated numeric/boolean generic subset. Every calculation includes a citation chain tracing values back to authoritative law.

## Installation

```bash
pip install rulespec-compile
```

## Guides

- `docs/compiler-architecture.md`: one-page architecture map, stability guide, and decision seams
- `docs/authoring-rulespec.md`: how to write `.yaml` files, external value rules, computed rules, imports, exports, and temporal definitions
- `docs/compile-and-lower.md`: CLI and Python API workflows for compile, lower, output selection, and rule binding
- `docs/validation-and-oracles.md`: harness, validation modes, execution modes, and current oracle lanes

## Quick start

### Command line

```bash
# Generate JavaScript EITC calculator
rulespec-compile eitc -o eitc.js

# Generate Python EITC calculator
rulespec-compile eitc --python -o eitc.py

# Compile a RuleSpec file to JavaScript
rulespec-compile compile examples/simple_tax.yaml -o simple_tax.js

# Compile a RuleSpec file to Python
rulespec-compile compile examples/snap.yaml --python -o snap.py

# Compile a RuleSpec file to Rust
rulespec-compile compile examples/snap.yaml --rust -o snap.rs

# Compile a real RuleSpec module and its local imports
rulespec-compile compile examples/working_families/benefit_amount.yaml --python -o benefit_amount.py

# Compile with import aliases to disambiguate duplicate module symbols
rulespec-compile compile examples/working_families/benefit_amount.yaml --python -o benefit_amount.py
# where benefit_amount.yaml contains lines like: import "./base_amount.yaml" as base

# Compile using explicit exports plus selective imports
rulespec-compile compile examples/working_families/base_amount.yaml --python -o base_amount.py
# where base_amount.yaml contains lines like:
# from "./phase_in_rate.yaml" import rate

# Compile with aliased public outputs
rulespec-compile compile examples/working_families/benefit_amount.yaml --python -o benefit_amount.py
# where benefit_amount.yaml contains lines like: export benefit as benefit_amount
# and --select-output uses the public name benefit_amount

# Re-export an imported symbol into a new public module surface
rulespec-compile compile examples/working_families/benefit_amount.yaml --python -o benefit_amount.py
# where benefit_amount.yaml contains lines like:
# export from "./base_amount.yaml" import base_amount

# Resolve bare imports through workspace module roots
rulespec-compile compile benefit_amount.yaml --python --module-root ./lib -o benefit_amount.py
# or configure rulespec.toml:
# [module_resolution]
# roots = ["./lib"]

# Resolve stable package-prefixed imports through workspace package aliases
rulespec-compile compile benefit_amount.yaml --python --package tax=./packages/tax -o benefit_amount.py
# or configure rulespec.toml:
# [module_resolution.packages]
# tax = "./packages/tax"

# Compile only the subgraph needed for one output
rulespec-compile compile examples/working_families/benefit_amount.yaml --binding phase_in_rate.rate=0.25 --select-output benefit_amount --python -o benefit_amount.py

# Emit the lowered selected-output bundle as JSON
rulespec-compile lower examples/working_families/benefit_amount.yaml --binding phase_in_rate.rate=0.25 --select-output benefit_amount -o benefit_amount.lowered.json

# Run the built-in compiler, batch-execution, and example-oracle scorecard
rulespec-compile harness

# Opt into curated live-stack checks against sibling RuleSpec files
# and Encoder artifacts
rulespec-compile harness --include-live

# Opt into external PolicyEngine-backed oracle checks (requires policyengine-us)
rulespec-compile harness --include-external

# Resolve temporal unified .yaml definitions for a specific date
rulespec-compile compile examples/snap.yaml --effective-date 2025-01-01 --python -o snap.py

# Bind a source-only external rule at compile time
rulespec-compile compile examples/working_families/base_amount.yaml --binding phase_in_rate.rate=0.25 --python -o base_amount.py
# or, for imported source-only rules, bind by rule identity:
rulespec-compile compile examples/working_families/benefit_amount.yaml --binding phase_in_rate.rate=0.25 --python -o benefit_amount.py

# Load rule bindings from a JSON file
rulespec-compile compile examples/working_families/benefit_amount.yaml --binding-file bindings.json --python -o benefit_amount.py

# Compile a current rulespec-us RuleSpec v1 module
rulespec-compile compile ../rulespec-us/statutes/26/3101/a.yaml \
  --select-output oasdi_wage_tax \
  --python -o oasdi_wage_tax.py

# Output to stdout
rulespec-compile eitc           # JavaScript
rulespec-compile eitc --python  # Python
```

### Python API

```python
from datetime import date
from pathlib import Path
from rulespec_compile import (
    generate_eitc_calculator_js,
    generate_eitc_calculator_py,
    load_rulespec_program,
)

# Pre-built EITC calculator (JavaScript)
js_code = generate_eitc_calculator_js()
print(js_code)

# Pre-built EITC calculator (Python)
py_code = generate_eitc_calculator_py()
print(py_code)

# Load and compile a real source-anchored RuleSpec file
program = load_rulespec_program(Path("examples/eitc.yaml"))

python_code = program.to_python_generator(
    effective_date=date(2025, 1, 1),
    outputs=["eitc"],
).generate()

lowered_json = program.to_lowered_program(
    effective_date=date(2025, 1, 1),
    outputs=["eitc"],
).to_json()

rust_code = program.to_rust_generator(
    effective_date=date(2025, 1, 1),
    outputs=["eitc"],
).generate()
```

For real policy work, keep RuleSpec in `.yaml` files with `source:` metadata and cited
rule / source metadata. The public docs prefer file-backed examples over
inline RuleSpec strings for that reason.

### Generated output

```javascript
const PARAMS = {
  credit_pct: { 0: 7.65, 1: 34, 2: 40, 3: 45 },  // 26 USC 32(b)(1)
  // ...
};

function calculate({ earned_income = 0, agi = 0, n_children = 0, is_joint = false }) {
  const eitc = /* formula */;

  return {
    eitc,
    citations: [
      {
        param: "credit_pct",
        module_identity: "eitc",
        source: "26 USC 32(b)(1)"
      },
      { variable: "eitc", module_identity: "eitc", source: "26 USC 32" },
    ],
  };
}

export { calculate, PARAMS };
export default calculate;
```

## Features

- **Multi-target compilation**: Generate JavaScript, Python, or Rust from the same DSL
- **Citation chains**: Every calculation traces back to statute/guidance
- **Zero dependencies**: Generated code runs standalone (JS in browsers, Python anywhere)
- **ESM exports**: JavaScript works with modern bundlers and `<script type="module">`
- **Type hints**: Python output includes full type annotations, TypeScript support coming soon

## Generic Compile Scope

The generic `rulespec-compile compile` path now shares one parsed compile model for JavaScript, Python, and Rust.

- Supported: straight-line formulas with assignments plus a final `return`
- Supported: scalar expressions built from arithmetic, comparisons, boolean operators, ternaries, indexed lookup access, inline RuleSpec conditionals like `if cond: a else: b`, and `abs` / `ceil` / `floor` / `max` / `min` / `round`
- Supported: limited `if` / `elif` / `else` formula blocks when every reachable path returns a value
- Supported: external numeric rule references discovered from parsed formulas, with free references exposed as calculator inputs
- Supported: inline numeric external rule values from `.yaml` `values:` blocks and single-entry temporal `.yaml` source rules, with exact integer-vs-number kinds preserved in the lowered bundle
- Supported: multi-entry temporal unified `.yaml` external values and formulas when `--effective-date` is provided
- Supported: source-only external rules when you bind them explicitly with `--binding NAME=VALUE`, `--binding module_identity.symbol=VALUE`, or indexed variants
- Supported: source-only external rules from repeated `--binding-file` inputs, including JSON/YAML rule-binding bundles, with inline `--binding` flags overriding file values
- Supported: explicit scalar-vs-indexed external lookup contracts in the lowered bundle, with bare rule references validated against resolved value shape
- Supported: output-focused compilation via repeated `--select-output NAME`, pruning to the reachable variable subgraph for those outputs
- Supported: lowered bundle emission via `rulespec-compile lower`, producing a serializable post-resolution artifact with explicit inputs, typed external values, typed ordered computations, and typed public outputs
- Supported: Rust output via `rulespec-compile compile ... --rust`, using the same lowered bundle as JS/Python for the validated numeric/boolean subset
- Supported: local file imports written as `import "./shared.yaml"` or `import "../common/base.yaml"`, with graph-wide reachability pruning from the selected outputs
- Supported: current `format: rulespec/v1` files with `rules:` entries, versioned formulas, and plural repository roots such as `statutes/...`, `regulations/...`, and `policies/...`
- Supported: spec-style top-level or per-rule `imports:` blocks using `path#symbol` syntax, including root-qualified paths like `statutes/...` and `regulations/...`
- Supported: bare imports like `from "tax/shared.yaml" import rate` when resolved through `rulespec.toml` module roots or repeated `--module-root DIR`
- Supported: stable workspace package aliases through `rulespec.toml` `[module_resolution.packages]` or repeated `--package NAME=DIR`
- Supported: import aliases written as `import "./shared.yaml" as shared`, with module-qualified references like `shared.rate`
- Supported: explicit module exports via `export tax, taxable_income`
- Supported: export aliases via `export taxable_income as base_income`
- Supported: module re-exports via `export from "./shared.yaml" import tax, rate as public_rate`
- Supported: selective imports via `from "./shared.yaml" import tax, threshold as income_threshold`
- Supported: entry-file output selection against the public export surface, including aliased output names
- Supported: first-class rule/module identity preserved through lowered bundles and generated citations; canonical `statutes/...`, `regulations/...`, and `policies/...` files use their path identity
- Supported: imported free inputs are exposed in lowered bundles and generated runtime interfaces as `module_identity.symbol` instead of merged internal names
- Unsupported: package registries, remote imports, nested namespace chains beyond `alias.value`, wildcard re-exports, loops, match/case, try/except, and other statement forms outside assignments, `if` / `elif` / `else`, and `return`
- Unsupported: attribute access, custom helper calls, slices, and other expression forms outside the validated scalar subset
- Unsupported: string formula literals in Rust output, and the prebuilt `rulespec-compile eitc` shortcut still only emits JavaScript or Python

If a file has multiple temporal entries and you do not supply an effective date, the compiler errors instead of guessing.
If a referenced external rule has no inline numeric values and you do not bind it explicitly, the compiler errors instead of inventing a placeholder.
If a bare external rule reference resolves to multiple indexed values, the compiler errors instead of silently taking index `0`.
If a control-flow formula does not return a value on every reachable path, the compiler errors instead of emitting `None` / `undefined`.
If plain imports expose the same symbol name more than once, the compiler errors instead of guessing which one you meant.
If a selective import asks for a name a module does not export, the compiler errors instead of treating it as an input.
If a file defines explicit exports, output selection and generated result keys use those public names instead of hidden internal helper names.
If a re-export asks for a name a dependency does not export, the compiler errors instead of silently omitting it.
If a bare import has no configured module root, or resolves to more than one file across roots, the compiler errors instead of guessing.
If a package-prefixed import names an unknown package alias, or a configured package alias points at no file, the compiler errors instead of falling back to a different root.
If a loaded program contains two different `.yaml` files with the same canonical rule identity, the compiler errors instead of inventing an ambiguous identity.
If a bare rule binding name matches more than one imported source-only rule, the compiler errors instead of guessing; bind it as `module_identity.symbol`.

Unsupported constructs fail with an explicit compiler error instead of generating misleading output.

### Lowered Bundle

`rulespec-compile lower` emits the compiler's backend-neutral bundle after import resolution,
temporal resolution, external rule binding, and selected-output pruning. The JSON payload
includes:

- explicit public inputs
- resolved external rule values and sources, each with an explicit `value_kind` plus `lookup_kind` metadata, `index_value_kind` for indexed tables, and the originating file's `module_identity`
- topologically ordered computations expressed in validated statement/expression IR, each with an explicit `value_kind`, typed local slot metadata, and `module_identity`
- public outputs mapped to the internal computation names they expose, each with an explicit `value_kind` and `module_identity`

JS, Python, and Rust generation now consume that same lowered bundle internally,
so the artifact is the exact seam between graph compilation and target-specific
rendering.

### Rule Binding Bundle Format

`--binding-file` accepts a structured bundle with schema/versioning, rule
identity, and optional effective dates:

```json
{
  "schema_version": 1,
  "metadata": {
    "name": "TY2025 external rule bindings"
  },
  "bindings": [
    {
      "module_identity": "phase_in_rate",
      "symbol": "rate",
      "effective_date": "2025-01-01",
      "value": 0.25,
      "source": "bundle://ty2025",
      "unit": "rate"
    },
    {
      "module_identity": "thresholds",
      "symbol": "phase_in_cap",
      "values": {
        "0": 10000,
        "1": 20000,
        "2": 30000
      },
      "source": "bundle://ty2025"
    }
  ]
}
```

The explicit `bindings` form is the file contract because it carries rule
identity, metadata, and optional effective dates. Repeated `--binding-file`
flags are merged in order.

### Compiler Harness

`rulespec-compile harness` runs a built-in scorecard of named compiler cases across the
current supported subset. It is intended to give future compiler work an explicit
objective target: improve the harness score by adding support, or add a new
failing case first and then make it pass.

- Use `rulespec-compile harness` for the human-readable scorecard
- Use `rulespec-compile harness --json` for machine-readable output
- Use repeated `--case NAME` to run a focused subset while developing a feature
- The harness now also validates Rust when `rustc` is available locally
- `rulespec-compile harness --include-live` opts into curated real-file checks against current sibling repos such as `rulespec-us`
- `rulespec-compile harness --include-external` opts into PolicyEngine-backed oracle cases when `policyengine-us` is installed

### Validation

`rulespec-validate --mode sample` now runs the shipped compiled `.yaml` examples in the
per-household RuleSpec lane before comparing them to PolicyEngine. The full vectorized
validation path now uses the lowered-program batch executor for the current
validated lowered subset, including limited `if` / `elif` / `else` statement
blocks, instead of the old handwritten RuleSpec formulas.
Validation reports label the current modes explicitly as
`compiled_example` for per-household sample validation and `compiled_batch`
for the full batch path.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=rulespec_compile
```

## See also

- [The Axiom Foundation](https://axiom-foundation.org) - Open infrastructure for encoded law
- [pe-compile](https://github.com/PolicyEngine/pe-compile) - Similar tool for PolicyEngine
