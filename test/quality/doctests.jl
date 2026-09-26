# The doctests of this package, in its docstrings and in the manual under `docs/src`, as the
# `Doctests` job of `CI.yml` runs them.
#
# Documenter evaluates a page's `@meta` block in `Main`, and a `@safetestset` file runs in a module
# of its own, so `SimpleSolvers` is imported into `Main` first.

using SimpleSolvers
using Documenter: DocMeta, doctest

@eval Main import SimpleSolvers

# The printing context of a doctest block is the sandbox module Documenter evaluates it in, not
# `Main`, and this `DocTestSetup` is evaluated into that sandbox. So a name exported by
# SimpleSolvers prints bare there and an internal one prints qualified, which is how the expected
# outputs are written — see `linearsolver`'s, where `LinearSolver` is bare and `PivotedLUCache`
# carries the module. Nothing has to be bound in `Main` for that.
DocMeta.setdocmeta!(SimpleSolvers, :DocTestSetup, :(using SimpleSolvers); recursive = true)

doctest(SimpleSolvers)
