using SafeTestsets

@safetestset "Gradients                                                                       " begin include("gradient_tests.jl") end
@safetestset "Jacobians                                                                       " begin include("jacobian_tests.jl") end
@safetestset "Hessians                                                                        " begin include("hessian_tests.jl") end
@safetestset "Univariate objectives                                                           " begin include("univariate_objectives.jl") end
@safetestset "Multivariate objectives                                                         " begin include("multivariate_objectives.jl") end
@safetestset "Linear Solvers                                                                  " begin include("linear_solvers.jl") end
@safetestset "Line Searches                                                                   " begin include("linesearch_test.jl") end
@safetestset "Nonlinear Solvers                                                               " begin include("nonlinear_solvers.jl") end
@safetestset "Optimizers                                                                      " begin include("optimizers.jl") end