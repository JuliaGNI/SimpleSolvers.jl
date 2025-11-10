"""
    DummyLinesearchState <: LinesearchState

Used for the *fixed point iterator* ([`PicardMethod`](@ref)).
"""
struct DummyLinesearchState{T} <: LinesearchState{T} end

DummyLinesearchState(T::DataType) = DummyLinesearchState{T}()