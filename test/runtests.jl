using SafeTestsets

@safetestset "Gradients                                                                       " begin include("gradient_tests.jl") end
@safetestset "Jacobians                                                                       " begin include("jacobian_tests.jl") end
@safetestset "Hessians                                                                        " begin include("hessian_tests.jl") end
@safetestset "Linear Solvers                                                                  " begin include("linear_solvers_tests.jl") end
@safetestset "Line Searches                                                                   " begin include("line_searches_tests.jl") end
@safetestset "Nonlinear Solvers                                                               " begin include("nonlinear_solvers_tests.jl") end
@safetestset "Optimizers                                                                      " begin include("optimizers_tests.jl") end
