"""
    NoLinesearchState <: LinesearchState

Used for the *fixed point iterator* ([`PicardMethod`](@ref)).
"""
struct NoLinesearchState{T} <: LinesearchState{T} end

NoLinesearchState(T::DataType) = NoLinesearchState{T}()