# JET.jl static analysis. See https://github.com/aviatesk/JET.jl.

using JET
using SimpleSolvers
using Test

@testset "JET report_package" begin
    @test_broken isempty(JET.get_reports(JET.report_package(SimpleSolvers; toplevel_logger = nothing))) # issue #196
end
