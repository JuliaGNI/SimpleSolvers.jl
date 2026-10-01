# JET.jl static analysis of the hot path. See https://github.com/aviatesk/JET.jl.
#
# The entry points are the functions that a test file asserts allocation free with `@allocated`:
# `factorize!` and `ldiv!` of `LapackLU`, `RecursiveLU`, `PivotedQR` and `SVDSolver`, `ldiv!` of
# `UmfpackLU` (`test/linear/linear_solvers.jl`), `report_linesearch_status` and
# `solve_with_status` (`test/linesearch/linesearch.jl`), and `solve!` with a state
# (`test/linesearch/linesearch.jl`, `test/nonlinear/nonlinear_solver.jl`). The bound on
# `solve_with_status` of `StrongWolfe` is 64 bytes, not zero. Each line runs
# `JET.report_opt` at the concrete argument types of one test call, and asserts that JET reports
# no runtime dispatch in a frame of SimpleSolvers, or of the extension that the call reaches.
#
# Each `@allocated` call gives one line. The method that it reaches also gets one line for each
# further element type that a test outside `test/quality/` passes to that method directly. An
# element type that reaches the method only through another function, such as `solve` or the
# `solve!` without a state, gets no line.
#
# JET does not support every Julia version (the `pre` and `nightly` jobs); there the file
# records one `@test_skip`.

using JET
using LinearAlgebra
using RecursiveFactorization
using SimpleSolvers
using SparseArrays
using Sparspak
using Test

using SimpleSolvers: BierlaireQuadratic, Quadratic, factorize!

const JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE

const RECURSIVE_EXT = Base.get_extension(SimpleSolvers, :SimpleSolversRecursiveFactorizationExt)
const SPARSPAK_EXT = Base.get_extension(SimpleSolvers, :SimpleSolversSparspakExt)

# A matrix of each element type, in the form that the solver takes.
dense(::Type{T}) where {T} = Matrix{T}(I, 4, 4) + ones(T, 4, 4)
function sparse_spd(::Type{T}) where {T}
    SparseMatrixCSC{T, Int}(sparse(Float64[4 1 0; 1 4 1; 0 1 4]))
end

# The argument types of `factorize!(solver, A)` and of `ldiv!(x, solver, b)`. `eltype` of a
# `LinearSolver` is `Any`, so the element type is passed.
factorize_types(solver, A) = (typeof(solver), typeof(A))
ldiv_types(solver, T) = (Vector{T}, typeof(solver), Vector{T})

# The merit of the line search tests, `make_linesearch_problem` of `test/linesearch/linesearch.jl`.
f(x) = x^2 - 1
g(x) = 2x
δx(x) = -g(x) / 2
function make_linesearch_problem(x₀::Number)
    _f(α, _) = f(SimpleSolvers.compute_new_iterate(x₀, α, δx(x₀)))
    _d(α, _) = g(SimpleSolvers.compute_new_iterate(x₀, α, δx(x₀))) * δx(x₀)
    LinesearchProblem{typeof(x₀)}(_f, _d)
end

# The residual of the nonlinear solve tests.
F(y, x, params) = y .= x .^ 2 .- 2

if JET_WORKS
    @testset "factorize! and ldiv! of the pivoted LU methods" begin
        # `@allocated`: `LapackLU` and `RecursiveLU` at Float64. Then the other element types
        # that the tests pass to the same two methods.
        for (method, T) in ((LapackLU(), Float64), (RecursiveLU(), Float64),
            (LapackLU(), Float32), (LapackLU(), ComplexF32), (LapackLU(), ComplexF64),
            (RecursiveLU(), Float32))
            A = dense(T)
            solver = LinearSolver(method, A)
            @test isempty(JET.get_reports(JET.report_opt(
                factorize!, factorize_types(solver, A);
                target_modules = (SimpleSolvers, RECURSIVE_EXT))))
            @test isempty(JET.get_reports(JET.report_opt(ldiv!, ldiv_types(solver, T);
                target_modules = (SimpleSolvers, RECURSIVE_EXT))))
        end
    end

    @testset "ldiv! of the sparse direct methods" begin
        # `@allocated`: `UmfpackLU` at Float64. Then the complex type of the `UmfpackLU` tests,
        # and the three element types that the `SparspakLU` tests pass to the same method.
        for (method, T) in ((UmfpackLU(), Float64), (UmfpackLU(), ComplexF64),
            (SparspakLU(), Float32), (SparspakLU(), BigFloat),
            (SparspakLU(), Rational{BigInt}))
            solver = LinearSolver(method, sparse_spd(T))
            @test isempty(JET.get_reports(JET.report_opt(ldiv!, ldiv_types(solver, T);
                target_modules = (SimpleSolvers, SPARSPAK_EXT))))
        end
    end

    @testset "factorize! and ldiv! of $(nameof(typeof(method)))" for method in (PivotedQR(),
        SVDSolver())
        # `@allocated`: five element types. Then ComplexF32 and BigFloat.
        for T in (Float64, Float32, Float16, ComplexF64, ComplexF16, ComplexF32, BigFloat)
            A = dense(T)
            solver = LinearSolver(method, A)
            @test isempty(JET.get_reports(JET.report_opt(
                factorize!, factorize_types(solver, A);
                target_modules = (SimpleSolvers,))))
            @test isempty(JET.get_reports(JET.report_opt(ldiv!, ldiv_types(solver, T);
                target_modules = (SimpleSolvers,))))
        end
    end

    @testset "report_linesearch_status" begin
        @test isempty(JET.get_reports(JET.report_opt(
            SimpleSolvers.report_linesearch_status,
            (LinesearchStatus{Float64}, Symbol, typeof(Options(Float64; verbosity = 0)));
            target_modules = (SimpleSolvers,))))
    end

    @testset "solve_with_status of $(nameof(M))" for M in (Static, Backtracking, Bisection,
        Quadratic, BierlaireQuadratic, StrongWolfe)
        # `@allocated`: with a ceiling and without one; `StrongWolfe` with no parameters argument.
        # `Backtracking(; expand = true)` has the type of `Backtracking()`, so it adds no line.
        ls = Linesearch(make_linesearch_problem(2.0), M(); verbosity = 0)
        params = M === StrongWolfe ? ((),) :
                 ((typeof((x = 2.0, αmax = 10.0)),), (typeof((x = 2.0,)),))
        for p in params
            @test isempty(JET.get_reports(JET.report_opt(solve_with_status,
                (typeof(ls), Float64, p...); target_modules = (SimpleSolvers,))))
        end
        # Float32 and Float16, the other element types of "the line search contract holds for
        # every method". The merit captures `one_T`, as that test does: a closure that captures
        # `T` infers `one(T)` as `Any` on Julia 1.11.
        for T in (Float32, Float16)
            one_T = one(T)
            lsT = Linesearch(LinesearchProblem{T}((α, _) -> one_T - α, (α, _) -> -one_T),
                M(T); verbosity = 0)
            @test isempty(JET.get_reports(JET.report_opt(
                solve_with_status, (typeof(lsT), T);
                target_modules = (SimpleSolvers,))))
        end
    end

    @testset "solve! of a nonlinear solver with a state" begin
        # `@allocated`: `NewtonSolver` with each line search, `PicardSolver` and `DogLegSolver`
        x = ones(3)
        solvers = Any[NewtonSolver(x, similar(x); F = F, linesearch = ls, verbosity = 0)
                      for ls in (Static(), Backtracking(), Backtracking(; expand = true),
            Bisection(), Quadratic(), BierlaireQuadratic())]
        push!(solvers, PicardSolver(x, F, similar(x); verbosity = 0))
        push!(solvers, DogLegSolver(x, F, similar(x); verbosity = 0))
        for s in solvers
            state = SolverState(s)
            @test isempty(JET.get_reports(JET.report_opt(solve!,
                (typeof(x), typeof(s), typeof(state)); target_modules = (SimpleSolvers,))))
        end
        # Float32, from the root-finding sweep of `test/nonlinear/nonlinear_solver.jl`
        x32 = ones(Float32, 3)
        s32 = NewtonSolver(
            x32, similar(x32); F = F, linesearch = Static(Float32), verbosity = 0)
        state32 = SolverState(s32)
        @test isempty(JET.get_reports(JET.report_opt(solve!,
            (typeof(x32), typeof(s32), typeof(state32)); target_modules = (SimpleSolvers,))))
    end
else
    @test_skip "JET does not work on this Julia version"  # aviatesk/JET.jl#681
end
