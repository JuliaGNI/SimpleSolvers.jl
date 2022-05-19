
struct LinesearchCache{T, AT <: AbstractArray{T}}
    x̄::AT
    x::AT
    δ::AT
    g::AT

    function LinesearchCache(x::AT) where {T, AT <: AbstractArray{T}}
        new{T,AT}(zero(x), zero(x), zero(x), zero(x))
    end
end

function update!(cache::LinesearchCache, x̄::AbstractVector, δ::AbstractVector)
    cache.x̄ .= x̄
    cache.x .= x̄ .+ δ
    cache.δ .= δ
    return cache
end

function update!(cache::LinesearchCache, x̄::AbstractVector, δ::AbstractVector, g::AbstractVector)
    update!(cache, x̄, δ)
    cache.g .= g
    return cache
end

function initialize!(cache::LinesearchCache, x::AbstractVector)
    cache.x̄ .= eltype(x)(NaN)
    cache.x .= eltype(x)(NaN)
    cache.δ .= eltype(x)(NaN)
    cache.g .= eltype(x)(NaN)
    return cache
end

"create univariate objective for linesearch algorithm"
function linesearch_objective(objective::MultivariateObjective, cache::LinesearchCache{T}) where {T}
    function f(α)
        cache.x .= cache.x̄ .+ α .* cache.δ
        value(objective, cache.x)
    end

    function d(α)
        cache.x .= cache.x̄ .+ α .* cache.δ
        dot(gradient!(objective, cache.x), cache.δ)
    end

    UnivariateObjective(f, d, one(T))
end
