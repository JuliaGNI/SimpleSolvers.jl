using SafeTestsets

# @safetestset "Print Statements                                                                " begin
#     include("check_print_statements.jl")
# end
@safetestset "Smoke Tests (construct every exported type)                                     " begin
    include("smoke_tests.jl")
end
@safetestset "Aqua Quality Assurance                                                          " begin
    include("aqua_tests.jl")
end
# @safetestset "JET Static Analysis                                                             " begin
#     include("jet_tests.jl")
# end
@safetestset "Gradients                                                                       " begin
    include("gradient_tests.jl")
end
@safetestset "Jacobians                                                                       " begin
    include("jacobian_tests.jl")
end
@safetestset "Nonlinear Problems                                                              " begin
    include("nonlinear_problem_tests.jl")
end
@safetestset "Neural Network Parameters                                                       " begin
    include("network_parameters_tests.jl")
end
@safetestset "Device Outer Product                                                           " begin
    include("device_outer.jl")
end
@safetestset "Hessians                                                                        " begin
    include("hessian_tests.jl")
end
@safetestset "Linear Solvers                                                                  " begin
    include("linear_solver_tests.jl")
end
@safetestset "Line Searches                                                                   " begin
    include("linesearch_tests.jl")
end
@safetestset "Nonlinear Solvers                                                               " begin
    include("nonlinear_solver_tests.jl")
end

# Doctests run in a local `Pkg.test()` but not across the CI test matrix: their output is
# architecture- and version-sensitive, and `CI.yml`'s pinned `Doctests` job is what checks them
# there. See `doctest.jl`. `SIMPLESOLVERS_DOCTESTS=true` forces them on anywhere.
if get(ENV, "SIMPLESOLVERS_DOCTESTS", "false") == "true" ||
   get(ENV, "CI", "false") != "true"
    @safetestset "Doctests                                                                        " begin
        include("doctest.jl")
    end
end
