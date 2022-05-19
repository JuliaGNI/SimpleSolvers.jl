
import LinearAlgebra: checksquare
import LinearAlgebra.BLAS: BlasFloat, BlasInt, liblapack, @blasfunc


struct LUSolverLAPACK{T<:BlasFloat} <: LinearSolver{T}
    n::BlasInt
    A::Matrix{T}
    pivots::Vector{BlasInt}
    info::BlasInt
end

function LUSolverLAPACK(A::Matrix{T}) where {T}
    n = checksquare(A)
    lu = LUSolverLAPACK{T}(n, zero(A), zeros(BlasInt, n), 0)
    factorize!(lu, A)
end

LUSolverLAPACK{T}(n::BlasInt) where {T} = LUSolverLAPACK(zeros(T, n, n))


## LAPACK LU factorization and solver for general matrices (GE)
for (getrf, getrs, elty) in
    ((:dgetrf_,:dgetrs_,:Float64),
     (:sgetrf_,:sgetrs_,:Float32),
     (:zgetrf_,:zgetrs_,:ComplexF64),
     (:cgetrf_,:cgetrs_,:ComplexF32))
    @eval begin
        function factorize!(lu::LUSolverLAPACK{$elty}, A::AbstractMatrix{$elty})
            copy!(lu.A, A)
            ccall((@blasfunc($getrf), liblapack), Nothing,
                  (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                   Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                   Ref(lu.n), Ref(lu.n), lu.A, Ref(lu.n), lu.pivots, Ref(lu.info))

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
                   Ref(trans), Ref(lu.n), Ref(nrhs), lu.A, Ref(lu.n), lu.pivots, x, Ref(lu.n), Ref(lu.info))

            if lu.info < 0
                throw(ArgumentError(lu.info))
            end
            return x
        end
    end
end
