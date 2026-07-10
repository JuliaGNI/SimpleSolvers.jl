# JET.jl static analysis (Phase 0.4, optional / allowed-to-warn).
#
# `report_package` surfaces potential runtime errors detected statically.  This
# set only reports and is not allowed to fail the suite — it is a diagnostic aid.
#
# JET is tightly coupled to Julia internals and regularly does not work on
# nightly / pre-release Julia versions (load or precompile failures inside
# JET/JuliaInterpreter).  Since the analysis is diagnostic-only, it is skipped
# gracefully when JET cannot load or run on the current Julia version.

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
