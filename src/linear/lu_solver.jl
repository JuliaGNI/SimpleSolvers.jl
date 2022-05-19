
mutable struct LUSolver{T} <: LinearSolver{T}
    n::Int
    A::Matrix{T}
    pivots::Vector{Int}
    perms::Vector{Int}
    info::Int
end

function LUSolver(A::AbstractMatrix{T}) where {T}
    n = checksquare(A)
    lu = LUSolver{T}(n, zero(A), zeros(Int, n), zeros(Int, n), 0)
    factorize!(lu, A)
end

LUSolver{T}(n::Int) where {T} = LUSolver(zeros(T, n, n))

function factorize!(lu::LUSolver{T}, A::AbstractMatrix{T}, pivot=true) where {T}
    copy!(lu.A, A)
    
    @inbounds for i in eachindex(lu.perms)
        lu.perms[i] = i
    end

    @inbounds for k in 1:lu.n
        # find index max
        kp = k
        if pivot
            amax = real(zero(T))
            for i in k:lu.n
                absi = abs(lu.A[i,k])
                if absi > amax
                    kp = i
                    amax = absi
                end
            end
        end
        lu.pivots[k] = kp
        lu.perms[k], lu.perms[kp] = lu.perms[kp], lu.perms[k]

        if lu.A[kp,k] != 0
            if k != kp
                # Interchange
                for i in 1:lu.n
                    tmp = lu.A[k,i]
                    lu.A[k,i] = lu.A[kp,i]
                    lu.A[kp,i] = tmp
                end
            end
            # Scale first column
            Akkinv = inv(lu.A[k,k])
            for i in k+1:lu.n
                lu.A[i,k] *= Akkinv
            end
        elseif lu.info == 0
            lu.info = k
        end
        # Update the rest
        for j in k+1:lu.n
            for i in k+1:lu.n
                lu.A[i,j] -= lu.A[i,k] * lu.A[k,j]
            end
        end
    end

    return lu
end

function LinearAlgebra.ldiv!(x::AbstractVector{T}, lu::LUSolver{T}, b::AbstractVector{T}) where {T}
    @assert axes(x,1) == axes(b,1) == axes(lu.A,1) == axes(lu.A,2)

    @inbounds for i in 1:lu.n
        x[i] = b[lu.perms[i]]
    end

    @inbounds for i in 2:lu.n
        s = zero(T)
        for j in 1:i-1
            s += lu.A[i,j] * x[j]
        end
        x[i] -= s
    end

    x[lu.n] /= lu.A[lu.n,lu.n]
    @inbounds for i in lu.n-1:-1:1
        s = zero(T)
        for j in i+1:lu.n
            s += lu.A[i,j] * x[j]
        end
        x[i] -= s
        x[i] /= lu.A[i,i]
    end

    return x
end
