"""
    NoLinesearch <: LinesearchMethod

Used for the *fixed point iterator* ([`PicardMethod`](@ref)).
"""
struct NoLinesearch{T} <: LinesearchMethod end

NoLinesearch(T::DataType) = NoLinesearch{T}()