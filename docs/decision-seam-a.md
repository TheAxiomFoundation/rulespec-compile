# Decision Seam A: unified external-rule resolver

**Status:** Draft — needs decision.

This document expands on Decision Seam A from
[`compiler-architecture.md`](compiler-architecture.md): *what artifact or
service should supply source-only external rule bindings for a given
`(module_identity, symbol, effective_date)`?*

The compiler already tracks the right rule identity end-to-end; what it still
lacks is a first-class resolver contract, and in its absence several ad-hoc
binding shapes have accumulated.

## Current state (ad hoc)

Today, external rule bindings reach the compiler through several paths that
were added incrementally and are not unified behind a single contract.

| Shape | Where | What it carries |
|-------|-------|-----------------|
| Python override maps | `compile_model` binding APIs | `{(identity, symbol): value}` dicts, typically populated from test fixtures |
| JSON override artifacts | CLI `--bindings` flag and harness fixtures | Flat `{identity: {symbol: value}}` objects, often checked in next to examples |
| YAML bundles | `rule_bindings.py` | Richer schema with metadata (source URL, effective date, notes) |
| Re-export resolution | `program.py` | Module-graph walks that resolve aliases before binding |

Consequences of the ad-hoc state:

- Identity resolution is duplicated — each loader re-implements "how do I look
  up `(module_identity, symbol, effective_date)` in this artifact?"
- Provenance leakage — JSON overrides carry no source metadata, so lowered
  bundles produced from them lose citation traceability.
- Effective-date handling is inconsistent — YAML bundles express ranges;
  Python/JSON maps typically carry only a single snapshot.
- Harness wiring is not testable against a contract — it tests against
  whatever artifact shape a given fixture happens to use.

## Name collision with existing concrete class

Before proposing a protocol name, note that `rule_bindings.RuleResolver`
already exists today as a concrete `@dataclass(frozen=True)` with a
**different** signature from what this seam needs. That class is used at
`isinstance(value, RuleResolver)` sites throughout the compiler — in
`compile_model`, `harness`, `cli`, the test suite, and in
`normalize_rule_bindings` / `merge_rule_bindings`. Adopting the name
`RuleResolver` for a new `Protocol` would either (a) shadow the existing
class and break every `isinstance` site, or (b) force a same-PR rename of the
existing class before design sign-off.

To keep this doc as a pure design proposal that does not require touching
live code, the new contract is named **`RuleSource`** below. The existing
`rule_bindings.RuleResolver` dataclass keeps its name and its current
behavior; if and when this seam lands, we can revisit whether to rename it
(e.g. to `StaticRuleResolver`) as a follow-up.

## Proposed unified resolver contract

A minimal `RuleSource` protocol that all sources conform to:

```python
class RuleSource(Protocol):
    def resolve(
        self,
        module_identity: ModuleIdentity,
        symbol: str,
        effective_date: date,
    ) -> ResolvedRule | None: ...

    def provenance(
        self,
        module_identity: ModuleIdentity,
        symbol: str,
    ) -> RuleProvenance: ...
```

Where `ResolvedRule` carries the value plus the effective interval it came
from, and `RuleProvenance` carries source citation, retrieval date, and
authoring metadata.

Adapters would wrap each current shape:

- `DictRuleSource` — wraps the current Python override maps
- `JsonBundleRuleSource` — wraps flat JSON artifacts, with an optional
  side-channel for injected provenance
- `YamlBundleRuleSource` — native mapping from the existing YAML schema
- `CompositeRuleSource` — ordered fallback across sources (e.g. test
  overrides → checked-in YAML → Axiom corpus service)

`compile_model` would depend only on the protocol. Everything else becomes an
adapter.

## Migration path

1. Land the `RuleSource` protocol and the three local adapters
   (`DictRuleSource`, `JsonBundleRuleSource`, `YamlBundleRuleSource`). No
   behavior change, and no rename of the existing
   `rule_bindings.RuleResolver` dataclass: all current call sites construct
   the appropriate adapter and pass it in alongside existing bindings.
2. Route the harness and CLI through the adapters. Remove the direct raw-dict
   paths from `compile_model`.
3. Add a `CompositeRuleSource` and wire the CLI `--bindings` flag to build
   one.
4. Introduce an `AxiomCorpusRuleSource` (or equivalent) that talks to an external
   source of truth, slotting in as the last adapter.
5. Once the Axiom corpus is the primary source, downgrade local JSON/YAML overrides to
   test-only scope. At that point, consider renaming the legacy
   `rule_bindings.RuleResolver` dataclass to `StaticRuleResolver` for
   clarity, since the `RuleSource` protocol is the primary contract.

## Open questions

- **Identity canonicalization.** How should the resolver treat
  `statute/us/26/32` vs `statute/us-federal/26/32`? Ties into Decision Seam B
  (rule-identity policy).
- **Effective-date algebra.** Do we want an explicit "no rule" result vs.
  `None`? Callers currently conflate the two.
- **Caching and determinism.** An external resolver needs a freeze mode for
  reproducible builds — probably a snapshot manifest committed alongside the
  lowered bundle.
- **Failure modes.** Today a missing binding raises at compile time. Should
  the resolver distinguish "not found" from "found, but value is null"?

## Next action

Before writing code: confirm protocol shape with a real Axiom corpus integration
sketch, so we don't land a contract that the first real external source cannot
satisfy.
