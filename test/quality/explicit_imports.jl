using ExplicitImports
using SimpleSolvers
using Sparspak
using Test

# Fails on an explicit import or a qualified access through a module other than the owner of the
# name, and on a self-qualified access. Sparspak is loaded so that the check sees
# `SimpleSolversSparspakExt` whatever ran before this file. Its stale import is marked broken
# below, so the stale-import check is off in `test_explicit_imports`.
test_explicit_imports(
    SimpleSolvers;
    no_stale_explicit_imports = false,
    # `src/SimpleSolvers.jl` loads `Distances`, `ForwardDiff`, `LinearAlgebra`, `Printf`,
    # `SparseArrays` and `StaticArrays` with a bare `using`
    no_implicit_imports = false,
    # `src/SimpleSolvers.jl` imports `SparseArrays.getcolptr` and `GeometricBase.update!`, which
    # their owners do not mark `public`
    all_explicit_imports_are_public = false,
    # `src/` reaches names that their owners do not mark `public`, such as
    # `LinearAlgebra.BlasFloat`, `LinearAlgebra.LAPACK.getrf!`, `ForwardDiff.GradientConfig` and
    # `Base.RefValue`
    all_qualified_accesses_are_public = false
)
@test_broken check_no_stale_explicit_imports(SimpleSolvers) === nothing  # #210
