# Smoke tests: construct every exported type with reasonable arguments.
#
# The point of this file is coverage of the *public constructors*, not numerical
# correctness — that is what would have caught the dead-on-arrival
# `LUSolverLAPACK` and generic `Jacobian` constructors.  All constructors that
# were once broken have been fixed; every check below is a plain `@test`.

using SimpleSolvers
using SparseArrays: SparseArrays
using SimpleSolvers: issufficient, isfloor, isconverged, isstalled, status
using Test

const T = Float64

# scalar-input objective and its derivatives (for gradients/hessians)
f_scalar(x::AbstractVector) = 1 + sum(x .^ 2)
∇f_scalar!(g::AbstractVector, x::AbstractVector) = (g .= 2 .* x; g)
Hf_scalar!(h::AbstractMatrix, x::AbstractVector) = (h .= 0; for i in eachindex(x)
        h[i, i] = 2
    end; h)

# vector-valued residual and its Jacobian (for jacobians/nonlinear problems)
F_vec!(y::AbstractVector, x::AbstractVector, params) = (y .= x .^ 2 .- 1; y)
J_vec!(j::AbstractMatrix, x::AbstractVector, params) = (j .= 0; for i in eachindex(x)
        j[i, i] = 2x[i]
    end; j)

const n = 2
const xvec = ones(T, n)
const Amat = T[4.0 5.0 -2.0; 7.0 -1.0 2.0; 3.0 1.0 4.0]
const yvec = T[1.0, 2.0, 3.0]

@testset "$(rpad("Abstract solver-method / state hierarchy", 80))" begin
    @test isabstracttype(SolverMethod)
    @test SolverState isa Function   # GeometricBase state constructor, not a type
    @test isabstracttype(NonlinearSolverMethod)
    @test isabstracttype(LinesearchMethod)
    # LinesearchMethod is a direct subtype of SolverMethod.
    @test LinesearchMethod <: SolverMethod
    @test NonlinearSolverMethod <: SolverMethod
    @test !(LinesearchMethod <: NonlinearSolverMethod)
    @test isabstracttype(DirectMethod)
    @test isabstracttype(Gradient)
    @test isabstracttype(Hessian)
    @test isabstracttype(Jacobian)
end

@testset "$(rpad("Nonlinear methods", 80))" begin
    @test Newton() isa Newton
    @test QuasiNewton() isa Newton
    @test QuasiNewton(5) isa Newton
    @test !isdefined(SimpleSolvers, :NewtonMethod)
    @test !isdefined(SimpleSolvers, :QuasiNewtonMethod)
    @test Picard() isa Picard
    @test !isdefined(SimpleSolvers, :PicardMethod)
    @test DogLeg() isa DogLeg

    # `Newton` is constructable by name, with an
    # optional `refactorize` argument; `QuasiNewton` is a convenience
    # constructor for a `Newton` with a quasi-Newton `refactorize` default.
    @test Newton() === Newton(1)
    @test Newton().refactorize == 1
    @test Newton(3).refactorize == 3
    @test QuasiNewton() === Newton(5)
    @test QuasiNewton().refactorize == 5

    @test DogLeg().refactorize == 1
    @test DogLeg(4).refactorize == 4
end

@testset "$(rpad("Gradients", 80))" begin
    @test GradientAutodiff{T}(f_scalar, n) isa GradientAutodiff
    @test GradientFiniteDifferences{T}(f_scalar, n) isa GradientFiniteDifferences
    @test GradientFunction{T}(f_scalar, ∇f_scalar!, n) isa GradientFunction
end

@testset "$(rpad("Hessians", 80))" begin
    @test HessianAutodiff{T}(f_scalar, n) isa HessianAutodiff
    @test HessianFunction{T}(Hf_scalar!, n) isa HessianFunction
end

@testset "$(rpad("Jacobians", 80))" begin
    @test JacobianAutodiff{T}(F_vec!, n, n) isa JacobianAutodiff
    @test JacobianFiniteDifferences{T}(F_vec!, n, n) isa JacobianFiniteDifferences
    @test JacobianFunction{T}(F_vec!, J_vec!) isa JacobianFunction

    # generic backend-selecting constructors
    @test Jacobian{T}(F_vec!, n) isa Jacobian
    @test Jacobian(F_vec!, xvec) isa Jacobian
    @test Jacobian(F_vec!, xvec, xvec) isa Jacobian
end

@testset "$(rpad("Linear problems and solvers", 80))" begin
    @test LinearProblem(Amat, yvec) isa LinearProblem
    @test LinearProblem(Amat) isa LinearProblem
    @test LinearProblem(yvec) isa LinearProblem
    @test LinearProblem{T}(3) isa LinearProblem
    @test LinearProblem{T}(3, 3) isa LinearProblem

    @test LU() isa LU
    @test LU(; static = false) isa LU
    @test LinearSolver(LU(), Amat) isa LinearSolver

    @test LapackLU() isa LapackLU
    @test LinearSolver(LapackLU(), Amat) isa LinearSolver

    # The extension-backed methods: the *type* is constructible whether or not the backend is
    # loaded, which is the point of defining it in `src/`. Building a `LinearSolver` needs the
    # extension, and is covered in `linear_solver_tests.jl`, which imports both backends.
    @test RecursiveLU() isa RecursiveLU
    @test SparspakLU() isa SparspakLU

    @test UmfpackLU() isa UmfpackLU
    @test LinearSolver(UmfpackLU(), SparseArrays.sparse(Amat)) isa LinearSolver
end

@testset "$(rpad("Line searches", 80))" begin
    @test Static() isa Static
    @test Backtracking() isa Backtracking
    @test Bisection() isa Bisection
    @test Quadratic() isa Quadratic
    @test BierlaireQuadratic() isa BierlaireQuadratic
    @test StrongWolfe() isa StrongWolfe

    ls_prob = LinesearchProblem{T}((α, params) -> α^2, (α, params) -> 2α)
    @test ls_prob isa LinesearchProblem
    @test Linesearch(ls_prob, Static()) isa Linesearch
    @test Linesearch(ls_prob) isa Linesearch
    @test SimpleSolvers.with_config(Linesearch(ls_prob), Options(T)) isa Linesearch

    # the status API is available for every line search method
    for m in (Static(), Backtracking(), Bisection(), Quadratic(), BierlaireQuadratic(), StrongWolfe())
        st = solve_with_status(Linesearch(ls_prob, m; verbosity = 0), one(T))
        @test st isa LinesearchStatus
        @test st.outcome isa LinesearchOutcome
        @test issufficient(st) isa Bool
        @test isfloor(st) isa Bool
    end
    @test LINESEARCH_DECREASED isa LinesearchOutcome
    @test LINESEARCH_FLOOR isa LinesearchOutcome
    @test LINESEARCH_EXHAUSTED isa LinesearchOutcome
    @test LINESEARCH_NO_DESCENT isa LinesearchOutcome
    @test LINESEARCH_STATIONARY isa LinesearchOutcome
    @test LINESEARCH_UNKNOWN isa LinesearchOutcome
end

@testset "$(rpad("Nonlinear problems, solvers and state", 80))" begin
    @test NonlinearProblem(F_vec!, xvec, xvec) isa NonlinearProblem
    @test NonlinearProblem(F_vec!, J_vec!, xvec, xvec) isa NonlinearProblem

    @test Options() isa Options
    @test Options(; linesearch_max_iterations = 10, max_stalls = 3) isa Options
    @test NonlinearSolverException("msg") isa NonlinearSolverException
    @test NonlinearSolverState(xvec) isa NonlinearSolverState

    yr = zero(xvec)
    @test NewtonSolver(xvec, yr; F = F_vec!) isa NonlinearSolver
    @test NewtonSolver(xvec, yr; F = F_vec!, refactorize = 5) isa NonlinearSolver
    @test PicardSolver(xvec, yr; F = F_vec!) isa NonlinearSolver
    @test DogLegSolver(xvec, yr; F = F_vec!) isa NonlinearSolver

    @test NonlinearSolver(Newton(), xvec, yr; F = F_vec!) isa NonlinearSolver

    # the `NonlinearProblem` forms, with and without the residual prototype
    nlprob = NonlinearProblem(F_vec!, xvec, yr)
    @test NewtonSolver(xvec, nlprob, yr) isa NonlinearSolver
    @test NewtonSolver(xvec, nlprob) isa NonlinearSolver
    @test PicardSolver(xvec, nlprob) isa NonlinearSolver
    @test DogLegSolver(xvec, nlprob) isa NonlinearSolver
    @test NonlinearSolver(Newton(), xvec, nlprob) isa NonlinearSolver

    # and the wrappers that build such a solver themselves
    @test solve!(copy(xvec), nlprob, Newton(); verbosity = 0) isa AbstractVector
    @test solve(xvec, nlprob, Newton(); verbosity = 0) isa AbstractVector
    @test solve_with_status!(copy(xvec), nlprob, Newton(); verbosity = 0) isa
          NonlinearSolverStatus

    ns = NewtonSolver(xvec, yr; F = F_vec!, verbosity = 0)
    nstate = SolverState(ns)
    @test status(ns, nstate) isa NonlinearSolverStatus
    @test isconverged(status(ns, nstate)) isa Bool
    @test isstalled(status(ns, nstate), SimpleSolvers.config(ns)) isa Bool
end
