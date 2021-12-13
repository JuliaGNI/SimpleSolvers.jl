
macro define(name, definition)
    quote
        macro $(esc(name))()
            esc($(Expr(:quote, definition)))
        end
    end
end


alloc_x(x::Number) = typeof(x)(NaN)
alloc_f(x::Number) = real(typeof(x))(NaN)
alloc_d(x::Number) = typeof(x)(NaN)

alloc_x(x::AbstractArray) = eltype(x)(NaN) .* x
alloc_f(x::AbstractArray) = real(eltype(x))(NaN)
alloc_g(x::AbstractArray) = eltype(x)(NaN) .* x
alloc_h(x::AbstractArray) = eltype(x)(NaN) .* x*x'
alloc_j(x::AbstractArray, f::AbstractArray) = eltype(x)(NaN) .* vec(f) .* vec(x)'


function L2norm(x)
    local l2::eltype(x) = 0
    for xᵢ in x
        l2 += xᵢ^2
    end
    l2
end

function l2norm(x)
    sqrt(L2norm(x))
end

function maxnorm(x)
    local r² = zero(eltype(x))
    @inbounds for xᵢ in x
        r² = max(r², xᵢ^2)
    end
    sqrt(r²)
end


function outer!(O, x, y)
    @assert axes(O,1) == axes(x,1)
    @assert axes(O,2) == axes(y,1)
    @inbounds @simd for i in axes(O, 1)
        for j in axes(O, 2)
            O[i,j] = x[i] * y[j]
        end
    end

end
