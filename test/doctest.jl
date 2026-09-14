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

# The printing context of a doctest block is the sandbox module Documenter evaluates it in, not
# `Main`, and this `DocTestSetup` is evaluated into that sandbox. So a name exported by
# SimpleSolvers prints bare there and an internal one prints qualified, which is how the expected
# outputs are written — see `linearsolver`'s, where `LinearSolver` is bare and `PivotedLUCache`
# carries the module. Nothing has to be bound in `Main` for that.
DocMeta.setdocmeta!(SimpleSolvers, :DocTestSetup, :(using SimpleSolvers); recursive = true)

doctest(SimpleSolvers; manual = false)
