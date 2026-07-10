# JET.jl static analysis (diagnostic, non-failing). See https://github.com/aviatesk/JET.jl.
# JET can fail to load on nightly/pre-release Julia, so it is skipped gracefully
# when it cannot run on the current version.

using SimpleSolvers
using Test

@testset "JET report_package (diagnostic, non-failing)" begin
    ok = try
        @eval using JET
        report = JET.report_package(SimpleSolvers; toplevel_logger=nothing)
        reports = JET.get_reports(report)
        if !isempty(reports)
            @info "JET reported $(length(reports)) potential issue(s); see the report below (diagnostic, not failing the suite)."
            show(report)
        end
        true
    catch e
        @info "Skipping JET analysis: JET failed to load or run on Julia $(VERSION)." exception = (e, catch_backtrace())
        true
    end
    @test ok
end
