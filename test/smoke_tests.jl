# Smoke tests: construct every exported type with reasonable arguments.
#
# The point of this file is coverage of the *public constructors*, not numerical
# correctness — that is what would have caught the dead-on-arrival constructors
# reported in the 2026-07-09 review (§1.2 `LUSolverLAPACK`, §1.6 generic
# `Jacobian`).  Constructors that are currently broken are marked `@test_broken`
# and are to be flipped to `@test` once the corresponding Phase 1 fix lands.

using SimpleSolvers
using Test

const T = Float64

# scalar-input objective and its derivatives (for gradients/hessians)
f_scalar(x::AbstractVector) = 1 + sum(x .^ 2)
∇f_scalar!(g::AbstractVector, x::AbstractVector) = (g .= 2 .* x; g)
Hf_scalar!(h::AbstractMatrix, x::AbstractVector) = (h .= 0; for i in eachindex(x); h[i, i] = 2; end; h)

# vector-valued residual and its Jacobian (for jacobians/nonlinear problems)
F_vec!(y::AbstractVector, x::AbstractVector, params) = (y .= x .^ 2 .- 1; y)
J_vec!(j::AbstractMatrix, x::AbstractVector, params) = (j .= 0; for i in eachindex(x); j[i, i] = 2x[i]; end; j)

const n = 2
const xvec = ones(T, n)
const Amat = T[4.0 5.0 -2.0; 7.0 -1.0 2.0; 3.0 1.0 4.0]
const yvec = T[1.0, 2.0, 3.0]


@testset "$(rpad("Abstract solver-method / state hierarchy", 80))" begin
    @test isabstracttype(SolverMethod)
    @test SolverState isa Function   # GeometricBase state constructor, not a type
    @test isabstracttype(NonlinearSolverMethod)
    @test isabstracttype(LinesearchMethod)
    # Phase 5: LinesearchMethod is now a direct subtype of SolverMethod (the former
    # `NonlinearMethod` supertype was removed — a line search is not itself a
    # nonlinear-solver method).
    @test LinesearchMethod <: SolverMethod
    @test NonlinearSolverMethod <: SolverMethod
    @test !(LinesearchMethod <: NonlinearSolverMethod)
    @test isabstracttype(DirectMethod)
    @test isabstracttype(Gradient)
    @test isabstracttype(Hessian)
    @test isabstracttype(Jacobian)
end


@testset "$(rpad("Nonlinear methods", 80))" begin
    @test NewtonMethod() isa NewtonMethod
    @test Newton() isa NewtonMethod
    @test QuasiNewtonMethod() isa NewtonMethod
    @test QuasiNewtonMethod(5) isa NewtonMethod
    @test PicardMethod() isa PicardMethod
    @test DogLeg() isa DogLeg

    # Phase 3.3 / §2.6: `NewtonMethod{true}` is now constructable by name, with an
    # optional `refactorize` argument (previously only `NewtonMethod()` and
    # `NewtonMethod{false}(...)` existed, so `NewtonMethod{true}(1)` threw).
    @test NewtonMethod() === NewtonMethod{true}(1)
    @test NewtonMethod{true}().refactorize == 1
    @test NewtonMethod{true}(3).refactorize == 3
    @test QuasiNewtonMethod().refactorize == 5
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

    # generic backend-selecting constructors (§1.6)
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
    @test LU(; static=false) isa LU
    @test LinearSolver(LU(), Amat) isa LinearSolver
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
end


@testset "$(rpad("Nonlinear problems, solvers and state", 80))" begin
    @test NonlinearProblem(F_vec!, xvec, xvec) isa NonlinearProblem
    @test NonlinearProblem(F_vec!, J_vec!, xvec, xvec) isa NonlinearProblem

    @test Options() isa Options
    @test NonlinearSolverException("msg") isa NonlinearSolverException
    @test NonlinearSolverState(xvec) isa NonlinearSolverState

    yr = zero(xvec)
    @test NewtonSolver(xvec, yr; F=F_vec!) isa NonlinearSolver
    @test QuasiNewtonSolver(xvec, yr; F=F_vec!) isa NonlinearSolver
    @test PicardSolver(xvec, yr; F=F_vec!) isa NonlinearSolver
    @test DogLegSolver(xvec, yr; F=F_vec!) isa NonlinearSolver

    @test NonlinearSolver(NewtonMethod(), xvec, yr; F=F_vec!) isa NonlinearSolver
end
