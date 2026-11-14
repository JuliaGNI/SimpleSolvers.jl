# Aqua.jl quality-assurance checks. See https://github.com/JuliaTesting/Aqua.jl.

using Aqua
using SimpleSolvers
using Test

# DIAGNOSTIC, never merged: rebuild Aqua 0.8.14's persistent-tasks wrapper
# (`precompile_wrapper`, src/persistent_tasks.jl) in the same process context, but with the
# precompile output visible and `strict = true`.
let Pkg = Aqua.Pkg
    println("::group::Aqua wrapper rebuild inside Pkg.test")
    @show Base.julia_cmd()
    @show get(ENV, "JULIA_LOAD_PATH", nothing)
    @show get(ENV, "JULIA_DEPOT_PATH", nothing)
    @show LOAD_PATH
    prev_project = Base.active_project()::String
    isdefined(Pkg, :respect_sysimage_versions) && Pkg.respect_sysimage_versions(false)
    try
        pkgdir = dirname(dirname(pathof(SimpleSolvers)))
        wrapperdir = tempname()
        wrappername = basename(wrapperdir)
        Pkg.generate(wrapperdir)
        Pkg.activate(wrapperdir)
        Pkg.develop(Pkg.PackageSpec(path = pkgdir))
        Pkg.status(; mode = Pkg.PKGMODE_MANIFEST)
        open(joinpath(wrapperdir, "src", wrappername * ".jl"), "w") do io
            println(io, "module $wrappername\nusing SimpleSolvers\nend")
        end
        cmd = `$(Base.julia_cmd()) --project=$wrapperdir -e 'push!(LOAD_PATH, "@stdlib"); using Pkg; Pkg.precompile(; strict = true)'`
        proc = run(ignorestatus(cmd), stdin, stdout, stderr)
        println("wrapper precompile exit code: ", proc.exitcode)
    finally
        isdefined(Pkg, :respect_sysimage_versions) && Pkg.respect_sysimage_versions(true)
        Pkg.activate(prev_project)
    end
    println("::endgroup::")
end

Aqua.test_all(SimpleSolvers)
