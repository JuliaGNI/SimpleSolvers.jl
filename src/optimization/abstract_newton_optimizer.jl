
struct NewtonOptimizerCache{T, AT <: AbstractArray{T}}
    x̄::AT
    x::AT
    δ::AT

    function NewtonOptimizerCache(x::AT) where {T, AT <: AbstractArray{T}}
        new{T,AT}(zero(x), zero(x), zero(x))
    end
end

function update!(cache::NewtonOptimizerCache, status::OptimizerStatus)
    copyto!(cache.x̄, status.x̄)
    copyto!(cache.x, status.x)
    copyto!(cache.δ, status.δ)
end
