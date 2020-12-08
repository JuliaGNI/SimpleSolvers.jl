
mutable struct LUSolver{T} <: LinearSolver{T}
    n::Int
    A::Matrix{T}
    b::Vector{T}
    x::Vector{T}
    pivots::Vector{Int}
    perms::Vector{Int}
    info::Int

    LUSolver{T}(n) where {T} = new(n, zeros(T, n, n), zeros(T, n), zeros(T, n), zeros(Int, n), zeros(Int, n), 0)
end

function LUSolver(A::Matrix{T}) where {T}
    n = checksquare(A)
    lu = LUSolver{eltype(A)}(n)
    lu.A .= A
    lu
end

function LUSolver(A::Matrix{T}, b::Vector{T}) where {T}
    n = checksquare(A)
    @assert n == length(b)
    lu = LUSolver{eltype(A)}(n)
    lu.A .= A
    lu.b .= b
    lu
end

function factorize!(lu::LUSolver{T}, pivot=true) where {T}
    A = lu.A

    begin
        @inbounds for i in eachindex(lu.perms)
            lu.perms[i] = i
        end

        @inbounds for k = 1:lu.n
            # find index max
            kp = k
            if pivot
                amax = real(zero(T))
                for i = k:lu.n
                    absi = abs(A[i,k])
                    if absi > amax
                        kp = i
                        amax = absi
                    end
                end
            end
            lu.pivots[k] = kp
            lu.perms[k], lu.perms[kp] = lu.perms[kp], lu.perms[k]

            if A[kp,k] != 0
                if k != kp
                    # Interchange
                    for i = 1:lu.n
                        tmp = A[k,i]
                        A[k,i] = A[kp,i]
                        A[kp,i] = tmp
                    end
                end
                # Scale first column
                Akkinv = inv(A[k,k])
                for i = k+1:lu.n
                    A[i,k] *= Akkinv
                end
            elseif lu.info == 0
                lu.info = k
            end
            # Update the rest
            for j = k+1:lu.n
                for i = k+1:lu.n
                    A[i,j] -= A[i,k]*A[k,j]
                end
            end
        end

        lu.info
    end
end

function solve!(lu::LUSolver{T}) where {T}
    local s::T

    @inbounds for i = 1:lu.n
        lu.x[i] = lu.b[lu.perms[i]]
    end

    @inbounds for i = 2:lu.n
        s = 0
        for j = 1:i-1
            s += lu.A[i,j] * lu.x[j]
        end
        lu.x[i] -= s
    end

    lu.x[lu.n] /= lu.A[lu.n,lu.n]
    @inbounds for i = lu.n-1:-1:1
        s = 0
        for j = i+1:lu.n
            s += lu.A[i,j] * lu.x[j]
        end
        lu.x[i] -= s
        lu.x[i] /= lu.A[i,i]
    end

    lu.b .= lu.x

    lu.x
end
