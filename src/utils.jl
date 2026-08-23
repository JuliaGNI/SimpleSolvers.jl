"""
    alloc_x(x)

Allocate `NaN`s of the size of `x`.
"""
alloc_x

"""
    alloc_g(x)

Allocate `NaN`s of the size of the gradient of `f` (with respect to `x`).
"""
alloc_g

"""
    alloc_h(x)

Allocate `NaN`s of the size of the Hessian of `f` (with respect to `x`).
"""
alloc_h

_nan(::Type{T}) where {T<:AbstractFloat} = T(NaN)
_nan(::Type{Complex{T}}) where {T<:AbstractFloat} = Complex{T}(T(NaN))
_nan(::Type{T}) where {T<:Number} = error("Cannot allocate NaN-initialized storage for element type $(T): only floating-point element types support NaN. Provide a floating-point input.")

alloc_x(x::Number) = _nan(typeof(x))

alloc_x(x::AbstractArray{T}) where {T<:Number} = _nan(T) .* x

alloc_g(x::AbstractArray{T}) where {T<:Number} = _nan(T) .* x
alloc_h(x::AbstractArray{T}) where {T<:Number} = _nan(T) .* x * x'
alloc_j(x::AbstractVector{T}, y::AbstractVector) where {T<:Number} = _nan(T) .* y * x'

"""
    alloc_rhs(A)

Allocate a dense right-hand-side/solution vector of length `size(A, 1)` for the matrix `A`.

Deliberately dense even when `A` is not. The obvious spelling, `alloc_x(A[:, 1])`, gives a
sparse vector for a `SparseMatrixCSC`, and `NaN * 0 != 0`, so the sparse broadcast has to
store every entry anyway — a sparse vector with no structural zeros, which is strictly worse
than the `Vector` it should have been. The right-hand side of a linear system is dense in
every caller here.
"""
alloc_rhs(A::AbstractMatrix{T}) where {T} = fill(_nan(T), size(A, 1))

"""
    fill_nan!(A)

Fill `A` with `NaN`s.

For a sparse matrix only the *stored* entries are filled. That is not a shortcut: for a sparse
Jacobian the sparsity pattern is structural information the linear solver depends on — the
ordering and symbolic factorization in a [`SparseFactorizationCache`](@ref) were computed for
one pattern — so clearing has to preserve it. `fill!(A, NaN)` would throw for a
`SparseMatrixCSC` anyway, since `NaN` is not the structural zero.

Used wherever a cache or problem is initialized or cleared: [`clear!(::LinearProblem)`](@ref)
and the `initialize!` methods of [`NonlinearSolverCache`](@ref) and [`DogLegCache`](@ref).
"""
fill_nan!(A::AbstractArray{T}) where {T} = fill!(A, _nan(T))
fill_nan!(A::SparseMatrixCSC{T}) where {T} = (fill!(nonzeros(A), _nan(T)); A)

"""
    zero_like(A)

A zeroed matrix with the same storage as `A` — and, for a sparse matrix, the same *pattern*.

`zero(::SparseMatrixCSC)` returns a matrix with no stored entries at all, which for a sparse
Jacobian buffer is not a zeroed Jacobian but an empty one: the pattern is structural
information the caller's `DF!` assembles into, and a `DF!` that writes only where the pattern
says it may would find nowhere to write. Used for the line search's private Jacobian buffer,
which has to be interchangeable with the solver's own.
"""
zero_like(A::AbstractMatrix) = zero(A)
zero_like(A::SparseMatrixCSC{T}) where {T} = (B = copy(A); fill!(nonzeros(B), zero(T)); B)

"""
    copy_matrix!(dest, src)

Copy `src` into `dest`, preserving `dest`'s storage.

The sparse-aware replacement for `dest .= src`. Sparse-to-sparse copies the stored values
only, and requires the two patterns to be *identical* — a differing pattern is an error rather
than a silent reallocation, because the solver cache built from `dest` holds an ordering and a
symbolic factorization computed for `dest`'s pattern, and quietly changing it underneath would
either be wrong or turn every step into a fresh `O(nnz log nnz)` ordering.
"""
copy_matrix!(dest::AbstractMatrix, src::AbstractMatrix) = copyto!(dest, src)

function copy_matrix!(dest::SparseMatrixCSC, src::SparseMatrixCSC)
    if getcolptr(dest) != getcolptr(src) || rowvals(dest) != rowvals(src)
        throw(ArgumentError(
            "the source matrix has a different sparsity pattern from the destination; the " *
            "linear solver's symbolic factorization was built for the destination's pattern. " *
            "Keep the Jacobian's pattern fixed across iterations, e.g. by assembling into a " *
            "matrix built once from the prototype."))
    end
    copyto!(nonzeros(dest), nonzeros(src))
    dest
end

# A sparse Jacobian assembled into a dense linear problem, or vice versa, is a plumbing
# mistake worth naming: `copyto!` would either densify or throw something obscure.
copy_matrix!(dest::SparseMatrixCSC, src::AbstractMatrix) = throw(ArgumentError(
    "cannot copy a $(typeof(src)) into a SparseMatrixCSC without inventing a sparsity " *
    "pattern; pass a sparse jacobian_prototype and assemble a sparse Jacobian into it"))

copy_matrix!(dest::AbstractMatrix, src::SparseMatrixCSC) = copyto!(dest, src)

"""
    add_to_diagonal!(A, α)

Add `α` to every diagonal entry of `A`, in place.

Returns immediately for `α == 0`, which is the default
(`SimpleSolvers.REGULARIZATION_FACTOR`), so the regularization step of
[`SimpleSolvers.maybe_refactorize!`](@ref) costs nothing unless it was asked for.

For a sparse matrix the diagonal has to be *structurally present*; a linear-index view into a
`SparseMatrixCSC`, as the dense path uses, would both be wrong for a structural zero and cost
`O(log nnz)` per entry. A missing diagonal entry is an error, because a regularized Jacobian
needs somewhere to put the shift.
"""
function add_to_diagonal!(A::AbstractMatrix, α)
    iszero(α) && return A
    @view(A[diagind(A)]) .+= α
    A
end

function add_to_diagonal!(A::SparseMatrixCSC, α)
    iszero(α) && return A
    rows, vals = rowvals(A), nonzeros(A)
    for j in axes(A, 2)
        col = nzrange(A, j)
        k = searchsortedfirst(@view(rows[col]), j) + first(col) - 1
        (k <= last(col) && rows[k] == j) || throw(ArgumentError(
            "the sparse matrix has no stored entry at the diagonal position ($j, $j), so a " *
            "regularization factor cannot be added there; include the diagonal in the " *
            "jacobian_prototype's sparsity pattern"))
        vals[k] += α
    end
    A
end


function outer!(O, x, y)
    @assert axes(O, 1) == axes(x, 1)
    @assert axes(O, 2) == axes(y, 1)
    @inbounds @simd for i in axes(O, 1)
        for j in axes(O, 2)
            O[i, j] = x[i] * y[j]
        end
    end

end


"""
    const WARNING_COUNTS

How often each *diagnosis* has been reported so far, keyed by the `Symbol` passed to
[`should_report!`](@ref). Process-global, and guarded by [`WARNING_LOCK`](@ref).

The number of keys is bounded by the number of message sites, so this does not grow without
limit. It is reset by [`reset_warning_counts!`](@ref).
"""
const WARNING_COUNTS = Dict{Symbol,Int}()

"Guards [`WARNING_COUNTS`](@ref); see [`should_report!`](@ref)."
const WARNING_LOCK = ReentrantLock()

@doc raw"""
    should_report!(key)

Count one occurrence of the diagnosis `key` and return `true` when it should be reported, i.e.
on its 1st, 2nd, 4th, 8th, 16th … occurrence. Used by [`nonlinear_solver_warnings`](@ref) in
place of the `maxlog` keyword of `@warn`.

# Extended help

## Why not `maxlog`

A `NonlinearSolver` reports at most once per [`solve!`](@ref), which is the right rate for a
caller that solves once and far too high a one for a caller that solves in a loop — a
time-stepping integrator asking for a tolerance its problem cannot attain would get the same
message at every step. `maxlog` bounds that, but it is keyed on the *source location* of the
`@warn`, so its budget is process-global and is never reset: once spent, the message is gone for
the remainder of the session, including for later solves of entirely different problems. A
genuine failure late in a long run was therefore silent.

## What the backoff promises, and what it does not

Doubling gives ``O(\log N)`` messages over ``N`` solves — a handful over a run of any length —
while never reaching a point beyond which nothing is ever said again. Be clear about the two
halves of that:

- a diagnosis appearing for the **first time** is reported at once, whichever solve it happens
  in, because its counter is still at zero. This is the case `maxlog` got wrong, and it is the
  one that matters: a run that was healthy for ten thousand steps and then was not says so.
- a diagnosis that **keeps repeating** is reported on solves 1, 2, 4, 8, 16 … of its own count.
  Occurrence 10 is silent; occurrence 16 is not. Every occurrence is *not* promised, and asking
  for that would be asking for the flood back.

Keys should therefore be as specific as the distinctions worth waking the user for — which is why
the caller folds the dominant [`LinesearchOutcome`](@ref) into the key of a stagnation report: a
solve that starts stagnating for a *new* reason is a new diagnosis, and is reported immediately
rather than inheriting the suppressed counter of the old one.

`verbosity = 0` remains the way to silence a solver completely, and
[`NonlinearSolverStatus`](@ref) remains the way to *act* on an outcome rather than read about it.
"""
function should_report!(key::Symbol)
    # The lock is what `maxlog`'s own unsynchronized `Dict` lacks. It costs nothing worth counting:
    # this is only ever reached once per solve, from a branch that has already decided there is
    # something to say.
    @lock WARNING_LOCK begin
        n = get(WARNING_COUNTS, key, 0) + 1
        WARNING_COUNTS[key] = n
        ispow2(n)
    end
end

"""
    reset_warning_counts!()

Forget every count kept by [`should_report!`](@ref), so that the next occurrence of each
diagnosis is reported again.

Mostly for tests: unlike `maxlog` — which `Test.TestLogger` sees straight through, since the
suppression happens in the logger — a message the backoff suppresses is never emitted at all, so a
test that asserts on a message has to start from a known count.
"""
reset_warning_counts!() = (@lock WARNING_LOCK empty!(WARNING_COUNTS); nothing)
