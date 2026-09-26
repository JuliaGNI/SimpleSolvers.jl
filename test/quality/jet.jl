# JET.jl static analysis. See https://github.com/aviatesk/JET.jl.

using JET
using SimpleSolvers
using Test

@testset "JET report_package" begin
    reports = JET.get_reports(JET.report_package(SimpleSolvers; toplevel_logger = nothing))
    @test_broken isempty(reports) # issue #196
end
