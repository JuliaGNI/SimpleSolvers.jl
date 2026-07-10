# Aqua.jl quality-assurance checks — acceptance gate for the remediation plan.
#
# Phase 1 fixed the undefined-exports check (§1.1 — removed `BracketingMethod`,
# `IterativeMethod` and the other exported-but-undefined names).
#
# Phase 6 resolved the last outstanding check (`ambiguities`): Phase 1 had already
# removed the `convert` ambiguity (§1.5), and a *pre-existing* `bisection` arity
# ambiguity — the interval form `bisection(f, αmin::T, αmax::T, params)`
# (bisection.jl:41) and the single-`α` form `bisection(f, α::T, params, config)`
# (bisection.jl:104) both matching `(f, ::T, ::T, ::Options)` — is now resolved by
# a disambiguating `bisection(f, αmin::T, αmax::T, config::Options)` method.
# `Aqua.test_all` therefore now runs entirely with Aqua's defaults.

using Aqua
using SimpleSolvers
using Test

Aqua.test_all(SimpleSolvers)
