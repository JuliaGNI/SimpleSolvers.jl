# The doctests embedded in this package's docstrings, run as part of the ordinary test suite.
#
# They are skipped inside the CI test matrix. Doctest output is sensitive to both the Julia version
# and the architecture — the last ULP of an `ldiv!` result differs between x86_64 and aarch64 — so a
# matrix entry that disagrees reports a real difference that is not a defect. `CI.yml` runs them
# once, on a pinned entry, as the `Doctests` required check. This file is what gives a local
# `Pkg.test()` the same signal. `SIMPLESOLVERS_DOCTESTS=true` forces them on anywhere, the CI
# matrix included.
#
# `manual = false` because the pages under `docs/src` need the documentation environment —
# CairoMakie and DocumenterCitations — which the test environment does not carry. The documentation
# build and the `Doctests` job are what check those.

using SimpleSolvers
using Documenter: DocMeta, doctest

# A doctest compares printed output, and a type prints unqualified only where its module is visible
# from the printing context, which is `Main`. The expected outputs are written that way — see
# `linearsolver`'s, where the exported `LinearSolver` is bare and the internal `PivotedLUCache`
# carries the module — because `docs/make.jl` runs `using SimpleSolvers` in `Main`. Run through
# `runtests.jl` this file lives inside a `@safetestset`, i.e. an anonymous module, so nothing binds
# the name in `Main` and every such name comes back qualified. Bind it explicitly.
@eval Main using SimpleSolvers

DocMeta.setdocmeta!(SimpleSolvers, :DocTestSetup, :(using SimpleSolvers); recursive = true)

doctest(SimpleSolvers; manual = false)
