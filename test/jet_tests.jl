# JET.jl static analysis (Phase 0.4, optional / allowed-to-warn).
#
# `report_package` surfaces potential runtime errors detected statically.  The
# current source has known issues (see `bugs.md`), so this set only reports and
# is not allowed to fail the suite — it is a diagnostic aid for Phases 1–4.

using JET
using SimpleSolvers
using Test

@testset "JET report_package (diagnostic, non-failing)" begin
    report = JET.report_package(SimpleSolvers; toplevel_logger=nothing)
    reports = JET.get_reports(report)
    if !isempty(reports)
        @info "JET reported $(length(reports)) potential issue(s); see the report below (not failing the suite in Phase 0)."
        show(report)
    end
    @test true
end
