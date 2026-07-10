# Aqua.jl quality-assurance checks — acceptance gate for the code-quality remediation.
#
# The undefined-exports check now passes (`BracketingMethod`, `IterativeMethod`
# and the other exported-but-undefined names were removed).
#
# The last outstanding check (`ambiguities`) is now resolved: the `convert`
# ambiguity was removed, and a *pre-existing* `bisection` arity
# ambiguity — the interval form `bisection(f, αmin::T, αmax::T, params)`
# (bisection.jl:41) and the single-`α` form `bisection(f, α::T, params, config)`
# (bisection.jl:104) both matching `(f, ::T, ::T, ::Options)` — is now resolved by
# a disambiguating `bisection(f, αmin::T, αmax::T, config::Options)` method.
# `Aqua.test_all` therefore now runs entirely with Aqua's defaults.

using Aqua
using SimpleSolvers
using Test

Aqua.test_all(SimpleSolvers)
