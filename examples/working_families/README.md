# Working Families Example Graph

This directory is the smallest shipped multi-file RuleSpec example in the repo.

It demonstrates:

- leaf-named `.yaml` modules
- selective imports
- import aliases
- re-exports
- exported output aliases
- qualified source-only external rule binding via `module_identity.symbol`

## Files

- `phase_in_rate.yaml`: source-only external rule exported as `rate`
- `phase_in_cap.yaml`: inline scalar rule exported as `cap`
- `base_amount.yaml`: imported helper variable exported as `base_amount`
- `benefit_amount.yaml`: entry file that re-exports `base_amount` and publishes
  `benefit_amount`

## Compile

```bash
rulespec-compile compile examples/working_families/benefit_amount.yaml \
  --python \
  --binding phase_in_rate.rate=0.25 \
  --select-output benefit_amount \
  -o benefit_amount.py
```

## Lower

```bash
rulespec-compile lower examples/working_families/benefit_amount.yaml \
  --binding phase_in_rate.rate=0.25 \
  --select-output benefit_amount \
  -o benefit_amount.lowered.json
```

## Expected behavior

With `earned_income=4000` and `has_qualifying_child=true`, the selected public
output `benefit_amount` resolves to `1000`.
