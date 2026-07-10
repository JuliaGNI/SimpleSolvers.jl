# Aqua.jl quality-assurance checks. See https://github.com/JuliaTesting/Aqua.jl.

using Aqua
using SimpleSolvers
using Test

Aqua.test_all(SimpleSolvers)
