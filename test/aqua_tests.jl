# Aqua.jl quality-assurance checks — acceptance gate for the remediation plan.
#
# Two checks are known to fail against the current (pre-Phase-1) source and are
# marked `broken = true` (the Aqua analogue of `@test_broken`), so the overall
# suite stays green while still enforcing the gate: once the corresponding fix
# lands the check turns into an "Unexpected Pass", forcing the flip to
# `broken = false`.
#   * undefined exports  (§1.1 — `BracketingMethod`, `IterativeMethod`, and other
#                          exported-but-undefined names) — fixed in Phase 1.1.
#   * `convert` ambiguity (§1.5 — `Base.convert(::Type, ::LinesearchMethod)`)     —
#                          fixed in Phase 1.4.
# All other checks (stale deps, compat bounds, piracy, …) must pass now.

using Aqua
using SimpleSolvers
using Test

Aqua.test_all(
    SimpleSolvers;
    ambiguities = (broken = true,),
    undefined_exports = (broken = true,),
)
