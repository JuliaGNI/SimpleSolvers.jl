using SafeTestsets

@safetestset "Print Statements                                                                " begin
    include("check_print_statements.jl")
end
@safetestset "Gradients                                                                       " begin
    include("gradient_tests.jl")
end
@safetestset "Jacobians                                                                       " begin
    include("jacobian_tests.jl")
end
@safetestset "Jacobians with parameter-dependent functions                                    " begin
    include("jacobian_for_parameter_dependent_functions_tests.jl")
end
@safetestset "Nonlinear Systems for parameter-dependent functions                             " begin
    include("nonlinear_systems_for_parameter_dependent_functions_tests.jl")
end
@safetestset "Hessians                                                                        " begin
    include("hessian_tests.jl")
end
@safetestset "Multivariate problems                                                           " begin
    include("multivariate_problems.jl")
end
@safetestset "Linear Solvers                                                                  " begin
    include("linear_solver_tests.jl")
end
@safetestset "Line Searches                                                                   " begin
    include("linesearch_test.jl")
end
@safetestset "Line Searches (second test)                                                     " begin
    include("linesearch_test_2.jl")
end
@safetestset "Nonlinear Solvers                                                               " begin
    include("nonlinear_solver_tests.jl")
end
@safetestset "Fixed point iterator                                                            " begin
    include("fixed_point_iterator_tests.jl")
end
@safetestset "Optimizers                                                                      " begin
    include("optimizer_tests.jl")
end
