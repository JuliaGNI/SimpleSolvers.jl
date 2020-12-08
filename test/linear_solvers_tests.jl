
using SimpleSolvers
using Test


struct LinearSolverTest{T} <: LinearSolver{T} end

test_solver = LinearSolverTest{Float64}()

@test_throws ErrorException factorize!(test_solver)
@test_throws ErrorException solve!(test_solver)


A = [[+4.  +5.  -2.]
     [+7.  -1.  +2.]
     [+3.  +1.  +4.]]
b = [-14., +42., +28.]
x = [+4., -4., +5.]


function test_lu_solver(solver, A, b, x)
    # for T in (Float64, ComplexF64, Float32, ComplexF32) # TODO
    for T in (Float64, ComplexF64)
        AT = convert(Array{T,2}, A)
        bT = convert(Array{T,1}, b)
        xT = convert(Array{T,1}, x)

        lu1 = solver(AT, bT)
        factorize!(lu1)
        solve!(lu1)
        @test lu1.b ≈ xT atol=1E-14

        lu2 = solver(AT)
        factorize!(lu2)
        lu2.b .= bT
        solve!(lu2)
        @test lu2.b ≈ xT atol=1E-14
    end
end

test_lu_solver(LUSolverLAPACK, A, b, x)
test_lu_solver(LUSolver, A, b, x)
