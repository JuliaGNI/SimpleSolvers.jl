"""
    NonlinearProblem

A `NonlinearProblem` describes ``F(x) = y``, where we want to solve for ``x`` and ``F`` is in nonlinear in general (also compare this to [`LinearProblem`](@ref) and [`OptimizerProblem`](@ref)).

!!! info
    `NonlinearProblem`s are used for *solvers* whereas `OptimizerProblem`s are their equivalent for *optimizers*.


# Keys
- `F`: accessed by calling `Function(nlp)`,
- `J::Union{Callable, Missing}`: accessed by calling `Jacobian(nlp)`,
"""
struct NonlinearProblem{T,TF<:Callable,TJ<:Union{Callable,Missing}} <: AbstractProblem
    F::TF
    J::TJ

    function NonlinearProblem(F::Callable, J::Union{Callable,Missing}, x::Tx, f::Tx=x) where {T,Tx<:AbstractArray{T}}
        new{T,typeof(F),typeof(J)}(F, J)
    end
end

NonlinearProblem{T}(F::Callable, J::Union{Callable,Missing}, n₁::Integer, n₂::Integer; kwargs...) where {T} = NonlinearProblem(F, J, zeros(T, n₁); kwargs...)
NonlinearProblem{T}(F::Callable, n₁::Integer, n₂::Integer; kwargs...) where {T} = NonlinearProblem{T}(F, missing, n₁, n₂)

@doc raw"""
    NonlinearProblem(F, x, f)

Set `jacobian` ``\gets`` `missing` and call the [`NonlinearProblem`](@ref) constructor.
"""
function NonlinearProblem(F::Callable, x::AbstractArray, f::AbstractArray)
    NonlinearProblem(F, missing, x, f)
end

"""
    value!(y, x, params)

Evaluate the [`NonlinearProblem`](@ref) at `x`.
"""
function value!(y::AbstractArray{T}, nlp::NonlinearProblem{T}, x::AbstractArray{T}, params::OptionalParameters) where {T}
    nlp.F(y, x, params)
end

# function value(nlp::NonlinearProblem{T}, x::AbstractVector{T}, params) where {T<:Number}
#     f = zero(value(nlp))
#     Function(nlp)(f, x, params)
#     f
# end

Base.Function(nlp::NonlinearProblem) = nlp.F


"""
    Jacobian(nlp::NonlinearProblem)

Return the *Jacobian function* stored in `nlp`.

!!! warning
    Note that this is different from the [`Jacobian`](@ref) used in the [`NonlinearSolver`](@ref)! There the [`Jacobian`](@ref) is a separate `struct`.
"""
Jacobian(nlp::NonlinearProblem) = nlp.J

function jacobian!(j::AbstractMatrix{T}, nlp::NonlinearProblem{T}, x::AbstractArray{T}, params) where {T}
    nlp.J(j, x, params)
end

function jacobian!(::AbstractMatrix{T}, ::NonlinearProblem{T,FT,Missing}, ::AbstractArray{T}, params) where {T,FT<:Callable}
    error("NonlinearSystem does not contain Jacobian.")
end
