
macro define(name, definition)
    quote
        macro $(esc(name))()
            esc($(Expr(:quote, definition)))
        end
    end
end


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
