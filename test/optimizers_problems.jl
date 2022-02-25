
function F(x)
    # 1 + sum(x.^2)
    y = one(eltype(x))
    for _x in x
        y += _x^2
    end
    return y
end

function ∇F!(g, x)
    g .= 2 .* x
end

function H!(g, x)
    g .= 0
    for i in eachindex(x)
        g[i,i] = 2
    end
end

