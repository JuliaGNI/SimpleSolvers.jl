using ExplicitImports
using SimpleSolvers
using Test

# Fails on a stale explicit import, and on an explicit import or a qualified access through a
# module other than the owner of the name.
test_explicit_imports(
    SimpleSolvers;
    # `src/SimpleSolvers.jl` loads `LinearAlgebra`, `ForwardDiff` and `Distances` with `using`
    no_implicit_imports = false,
    # `src/SimpleSolvers.jl` imports `SparseArrays.getcolptr` and `GeometricBase.update!`, which
    # their owners do not mark `public`
    all_explicit_imports_are_public = false,
    # `src/linear/` reaches `LinearAlgebra.BlasFloat` and `src/base/` reaches
    # `ForwardDiff.GradientConfig`, which their owners do not mark `public`
    all_qualified_accesses_are_public = false
)
