"""
    DFPState

This is equivalent to [`BFGSState`](@ref).
"""
const DFPState = BFGSState

OptimizerState(::DFP, x_args...) = DFPState(x_args...)