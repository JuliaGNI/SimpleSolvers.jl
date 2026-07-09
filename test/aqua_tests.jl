# Aqua.jl quality-assurance checks — acceptance gate for the remediation plan.
#
# Phase 1 fixed the undefined-exports check (§1.1 — removed `BracketingMethod`,
# `IterativeMethod` and the other exported-but-undefined names), so that override
# is gone and the check now runs (and passes) with Aqua's defaults.
#
# One check remains `broken = true`:
#   * ambiguities — Phase 1 removed the `convert` ambiguity (§1.5, replaced the
#     `Base.convert(::Type, ::LinesearchMethod)` catch-all with `change_precision`),
#     but a *pre-existing* `bisection` arity ambiguity remains: the bracket form
#     `bisection(f, αmin::T, αmax::T, params)` (bisection.jl:41) and the single-`x`
#     form `bisection(f, α::T, params, config::Options)` (bisection.jl:81) both
#     match `(f, ::T, ::T, ::Options)`. Resolving it means reworking bisection's
#     overload set, which belongs to the Phase 2/4 bisection hardening — not the
#     mechanical Phase 1 sweep. Flip this to `broken = false` once that lands.

using Aqua
using SimpleSolvers
using Test

Aqua.test_all(
    SimpleSolvers;
    ambiguities = (broken = true,),
)
