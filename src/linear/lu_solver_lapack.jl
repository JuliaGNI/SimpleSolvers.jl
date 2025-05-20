import LinearAlgebra: checksquare
import LinearAlgebra.BLAS: BlasFloat, BlasInt, liblapack, @blasfunc

"""
    LUSolverLAPACK <: LinearSolver

The LU Solver taken from `LinearAlgebra.BLAS`.
"""
struct LUSolverLAPACK{T <: BlasFloat, LST <: LinearSystem{T}} <: LinearSolver{T}
    n::BlasInt
    linearsystem::LST
    pivots::Vector{BlasInt}
    info::BlasInt
end

linearsystem(lu::LUSolverLAPACK) = lu.linearsystem

function LUSolverLAPACK(ls::LST) where {T, LST <: LinearSystem{T}}
    n = checksquare(Matrix(ls))
    lu = LUSolverLAPACK{T, LST}(n, ls, zeros(BlasInt, n), BlasInt(0))
    solve!(lu)
end

function LUSolverLAPACK{T}(n::BlasInt) where {T}
    ls = LinearSystem{T}(n)
    LUSolverLAPACK(ls)
end

function LUSolverLAPACK(A::AbstractMatrix)
    LUSolverLAPACK(LinearSystem(A))
end

## LAPACK LU factorization and solver for general matrices (GE)
for (getrf, getrs, elty) in
    ((:dgetrf_,:dgetrs_,:Float64),
     (:sgetrf_,:sgetrs_,:Float32),
     (:zgetrf_,:zgetrs_,:ComplexF64),
     (:cgetrf_,:cgetrs_,:ComplexF32))
    @eval begin
        function factorize!(lu::LUSolverLAPACK{$elty}, A::AbstractMatrix{$elty})
            copy!(Matrix(linearsystem(lu)), A)
            ccall((@blasfunc($getrf), liblapack), Nothing,
                  (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                   Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                   Ref(lu.n), Ref(lu.n), Matrix(linearsystem(lu)), Ref(lu.n), lu.pivots, Ref(lu.info))

            if lu.info > 0
                throw(SingularException(lu.info))
            elseif lu.info < 0
                throw(ArgumentError(lu.info))
            end
            return lu
        end

        function LinearAlgebra.ldiv!(x::AbstractVector{$elty}, lu::LUSolverLAPACK{$elty}, b::AbstractVector{$elty})
            copy!(x, b)
            trans = UInt8('N')
            nrhs = BlasInt(1)
            ccall((@blasfunc($getrs), liblapack), Nothing,
                  (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                   Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                   Ref(trans), Ref(lu.n), Ref(nrhs), Matrix(linearsystem(lu)), Ref(lu.n), lu.pivots, x, Ref(lu.n), Ref(lu.info))

            if lu.info < 0
                throw(ArgumentError(lu.info))
            end
            return x
        end
    end
end

solution(lu::LUSolverLAPACK) = solution(linearsystem(lu))

function solve!(lu::LUSolverLAPACK)
    !status(linearsystem(lu)) || error("System has already been solved.")
    factorize!(lu)
    ldiv!(solution(lu), lu, rhs(linearsystem(lu)))
    linearsystem(lu).solved = true
    lu
end

factorize!(lu::LUSolverLAPACK) = factorize!(lu, Matrix(linearsystem(lu)))