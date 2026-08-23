module SimpleSolversSparspakExt

using LinearAlgebra: LinearAlgebra, SingularException
using SparseArrays: SparseMatrixCSC
using Sparspak: Sparspak
using Sparspak.SparseCSCInterface: sparspaklu, sparspaklu!

using SimpleSolvers: SparspakLU, LinearSolver, SparseFactorizationCache,
    cache, checkpattern, checksparse
import SimpleSolvers: LinearSolverCache, factorize!, _sparse_ldiv!, checkfactorized

function LinearSolverCache(method::SparspakLU, A::AbstractMatrix{T}) where {T}
    n = checksparse(method, A)
    # `factorize=false`: the LinearSolver constructor documents that it allocates but does not
    # factorize. This still does the ordering and symbolic factorization, which is exactly the
    # work we want done once and reused.
    SparseFactorizationCache{T}(sparspaklu(A; factorize=false), n)
end

# Sparspak's own `_factordone` is the authoritative flag — it is what `sparspaklu!` sets — so
# read it rather than keeping a second copy in the cache that could drift out of step.
function checkfactorized(lsolver::LinearSolver{T,SparspakLU}) where {T}
    cache(lsolver).F._factordone || throw(ArgumentError(
        "the SparspakLU solver has not been factorized yet; call factorize! before ldiv!/solve!."))
    nothing
end

"""
    factorize!(lsolver::LinearSolver{T,SparspakLU}, A)

Refactorize `A`, reusing the ordering and symbolic factorization already in the cache.

`allow_pattern_change=false` is deliberate: for the Newton loop this method exists to serve,
the pattern is fixed, and silently re-running the ordering would be a performance cliff rather
than a convenience.
"""
function factorize!(lsolver::LinearSolver{T,SparspakLU}, A::AbstractMatrix{T}) where {T}
    checkpattern(lsolver, A)
    c = cache(lsolver)
    sparspaklu!(c.F, A; allow_pattern_change=false)
    # Sparspak cannot tell us; `_sparse_ldiv!` finds out the hard way. See the docstring.
    c.info = 0
    c.factorized = true
    lsolver
end

# Sparspak factorizes a singular matrix without complaint and then returns non-finite numbers,
# so this is where singularity is detected. `SingularException(0)` rather than an index because
# there is no index to report.
function _sparse_ldiv!(::SparspakLU, c::SparseFactorizationCache{T}, x::AbstractVector{T}, b::AbstractVector{T}) where {T}
    LinearAlgebra.ldiv!(x, c.F, b)
    if !all(isfinite, x)
        c.info = 1
        throw(SingularException(0))
    end
    x
end

end # module
