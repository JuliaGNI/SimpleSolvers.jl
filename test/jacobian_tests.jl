using SimpleSolvers
using Test


@testset "Basic Jacobian functionality and consistency" begin

    n = 1
    T = Float64
    x = [T(π),]
    j = reshape(2x, 1, 1)


    function F!(f::AbstractVector, x::AbstractVector, params)
        f .= x .^ 2
    end

    function J!(g::AbstractMatrix, x::AbstractVector, params)
        g .= 0
        for i in eachindex(x)
            g[i, i] = 2x[i]
        end
        g
    end


    JPAD = JacobianAutodiff{T}(F!, n, n)
    JPFD = JacobianFiniteDifferences{T}(F!, n, n)
    JPUS = JacobianFunction{T}(F!, J!)

    @test typeof(JPAD) <: JacobianAutodiff
    @test typeof(JPFD) <: JacobianFiniteDifferences
    @test typeof(JPUS) <: JacobianFunction

    @test JPAD == JacobianAutodiff{T}(F!, n)
    @test JPAD == JacobianAutodiff(F!, x)
    @test JPFD == JacobianFiniteDifferences{T}(F!, n)
    @test JPFD == JacobianFiniteDifferences(F!, x)


    jad = zero(j)
    jfd = zero(j)
    jus = zero(j)

    JPAD(jad, x, nothing)
    JPFD(jfd, x, nothing)
    JPUS(jus, x, nothing)

    @test jad ≈ j atol = eps()
    @test jfd ≈ j atol = 1E-7
    @test jus == j


    jad1 = zero(j)
    jfd1 = zero(j)
    jus1 = zero(j)

    JPAD(jad1, x, nothing)
    JPFD(jfd1, x, nothing)
    JPUS(jus1, x, nothing)

    @test jad1 == jad
    @test jfd1 == jfd
    @test jus1 == jus


    jad2 = zero(j)
    jfd2 = zero(j)
    jus2 = zero(j)

    JPAD(jad2, x, nothing)
    JPFD(jfd2, x, nothing)
    JPUS(jus2, x, nothing)

    @test jad2 == jad
    @test jfd2 == jfd
    @test jus2 == jus

end


@testset "Jacobians with parameter-dependent functions" begin

    function F(f::AbstractVector{T}, x::AbstractVector{T}, params) where {T}
        f .= (params.A * x + params.b) .^ 2
    end

    function DF!(jacobian_matrix::AbstractMatrix{T}, x::AbstractVector{T}, params) where {T}
        for i in axes(jacobian_matrix, 1)
            for j in axes(jacobian_matrix, 2)
                jacobian_matrix[j, i] = 2 * params.A[j, i] * (params.A[j, :]' * x + params.b[j])
            end
        end
        jacobian_matrix
    end

    A₁ = [3.0 6.0 7.0; 9.0 18.0 19.0; 11.0 22.0 23.0]
    b₁ = [1.0, 1.0, 2.0]

    A₂ = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    b₂ = [1.0, 1.0, 1.0]

    jac₁ = JacobianAutodiff{eltype(A₁)}(F, size(A₁)[2], size(A₁)[1])
    jac₂ = JacobianFiniteDifferences{eltype(A₁)}(F, size(A₁)[2], size(A₁)[1])
    jac₃ = JacobianFunction{eltype(A₁)}(F, DF!)

    function test_various_jacobians(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
        params = (A=A, b=b)
        x = rand(T, length(A[1, :]))
        j₁ = zero(A)
        j₂ = zero(A)
        j₃ = zero(A)

        @test jac₁(j₁, x, params) ≈ jac₂(j₂, x, params) ≈ jac₃(j₃, x, params)
    end

    for (A, b) in ((A₁, b₁), (A₂, b₂))
        test_various_jacobians(A, b)
    end

end


# Regression: the generic backend-selecting `Jacobian` constructors used to
# forward to a nonexistent `Jacobian{T}(F, nx, ny)` method and always threw. They
# dispatch to `JacobianAutodiff` / `JacobianFiniteDifferences` via `mode`.
@testset "Generic Jacobian backend selection" begin
    T = Float64
    F!(f::AbstractVector, x::AbstractVector, params) = (f .= x .^ 2; f)
    x = T[1.0, 2.0]
    jref = [2.0 0.0; 0.0 4.0]

    # default backend is autodiff
    @test Jacobian{T}(F!, 2, 2) isa JacobianAutodiff
    @test Jacobian{T}(F!, 2) isa JacobianAutodiff
    @test Jacobian(F!, x) isa JacobianAutodiff
    @test Jacobian(F!, x, x) isa JacobianAutodiff

    # explicit backend selection
    @test Jacobian{T}(F!, 2, 2; mode=:autodiff) isa JacobianAutodiff
    @test Jacobian{T}(F!, 2, 2; mode=:finitedifferences) isa JacobianFiniteDifferences
    @test_throws ErrorException Jacobian{T}(F!, 2, 2; mode=:nonsense)

    # both backends compute the correct Jacobian
    jad = Jacobian{T}(F!, 2, 2; mode=:autodiff)
    jfd = Jacobian{T}(F!, 2, 2; mode=:finitedifferences)
    mad = zeros(T, 2, 2)
    mfd = zeros(T, 2, 2)
    jad(mad, x, nothing)
    jfd(mfd, x, nothing)
    @test mad ≈ jref atol = eps()
    @test mfd ≈ jref atol = 1e-7
end


# Regression: the finite-difference Jacobian functor used to iterate its row
# loop over `eachindex(x)` (input indices) instead of the output indices. For a
# non-square Jacobian this silently left rows unwritten (`ny > nx`) or threw a
# `BoundsError` (`ny < nx`).  It loops over `eachindex(jac.f1)` (outputs).
@testset "Finite-difference non-square Jacobians" begin
    T = Float64

    # 2×3 Jacobian (ny = 2 < nx = 3)
    F23(f, x, params) = (f[1] = x[1] + 2x[2] + 3x[3]; f[2] = 4x[1] + 5x[2] + 6x[3]; f)
    J23 = T[1 2 3; 4 5 6]
    x3 = T[1.0, 2.0, 3.0]
    jfd23 = JacobianFiniteDifferences{T}(F23, 3, 2)
    jad23 = JacobianAutodiff{T}(F23, 3, 2)
    mfd23 = zeros(T, 2, 3)
    mad23 = zeros(T, 2, 3)
    jfd23(mfd23, x3, nothing)
    jad23(mad23, x3, nothing)
    @test mfd23 ≈ J23 atol = 1e-6
    @test mad23 ≈ J23 atol = eps()

    # 3×2 Jacobian (ny = 3 > nx = 2)
    F32(f, x, params) = (f[1] = x[1]; f[2] = x[2]; f[3] = x[1] * x[2]; f)
    x2 = T[3.0, 4.0]
    J32 = T[1 0; 0 1; x2[2] x2[1]]
    jfd32 = JacobianFiniteDifferences{T}(F32, 2, 3)
    jad32 = JacobianAutodiff{T}(F32, 2, 3)
    mfd32 = zeros(T, 3, 2)
    mad32 = zeros(T, 3, 2)
    jfd32(mfd32, x2, nothing)
    jad32(mad32, x2, nothing)
    @test mfd32 ≈ J32 atol = 1e-6
    @test mad32 ≈ J32 atol = eps()
end

# Copilot review finding on PR #161: the JacobianAutodiff signature check used
# `hasmethod(F, Tuple{typeof(y), typeof(x), Any})`, which only matches methods
# accepting *arbitrary* params — a valid `F(y, x, params::MyParams)` with a
# concretely typed params argument was spuriously rejected.  The check uses
# `methods` (type intersection), which accepts any 3-argument form matching
# (y, x) while still rejecting functions without a params argument.
struct JacTestParams end

@testset "JacobianAutodiff accepts params-typed functions" begin
    Ftyped!(y, x, params::JacTestParams) = (y .= x .^ 2)
    jac = JacobianAutodiff(Ftyped!, rand(2), rand(2))
    @test jac isa JacobianAutodiff
    J = zeros(2, 2)
    x = [1.0, 2.0]
    jac(J, x, JacTestParams())
    @test J ≈ [2.0 0.0; 0.0 4.0] atol = 10eps()

    # a function without a params argument is still rejected with a clear error
    F2arg!(y, x) = (y .= x .^ 2)
    @test_throws ErrorException JacobianAutodiff(F2arg!, rand(2), rand(2))
end

# The exported `check_jacobian` / `print_jacobian` diagnostics (matrix forms)
# write to an `io` argument, so their output is captured with `sprint` and checked.
@testset "check_jacobian / print_jacobian (matrix forms) write correct output" begin
    J = [1.0 √2.0; √2.0 3.0]                      # cond = 7+4√3 ≈ 13.9282, det = 1
    out = replace(sprint(check_jacobian, J), r"[ \t]+" => " ")
    @test occursin("Condition Number of Jacobian: 13.9282", out)
    @test occursin("Determinant of Jacobian: 1.0", out)
    @test occursin("minimum(|Jacobian|): 1.0", out)
    @test occursin("maximum(|Jacobian|): 3.0", out)
    # the digits keyword is honoured
    out2 = replace(sprint(io -> check_jacobian(io, J; digits=2)), r"[ \t]+" => " ")
    @test occursin("Condition Number of Jacobian: 13.93", out2)

    # print_jacobian reproduces the text/plain table exactly (aligned, with header)
    @test sprint(print_jacobian, J) == repr("text/plain", J) * "\n"
    @test occursin("Matrix{Float64}", sprint(print_jacobian, J))

    # the convenience forms without `io` write to stdout and forward keywords
    # (called silently; content is covered by the `sprint` checks above)
    @test redirect_stdout(() -> check_jacobian(J), devnull) === nothing
    @test redirect_stdout(() -> check_jacobian(J; digits=2), devnull) === nothing
    @test redirect_stdout(() -> print_jacobian(J), devnull) === nothing
end
