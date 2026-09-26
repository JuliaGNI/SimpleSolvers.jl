using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "JET" include("quality/jet.jl")
    @safetestset "Smoke tests" include("integration/smoke.jl")
    @safetestset "Gradients" include("base/gradient.jl")
    @safetestset "Jacobians" include("base/jacobian.jl")
    @safetestset "Hessians" include("base/hessian.jl")
    @safetestset "Outer product on a device" include("utils.jl")
    @safetestset "Neural network parameters" include("integration/neural_network_parameters_ext.jl")
    @safetestset "Linear solvers" include("linear/linear_solvers.jl")
    @safetestset "Line searches" include("linesearch/linesearch.jl")
    @safetestset "Nonlinear problems" include("nonlinear/nonlinear_problem.jl")
    @safetestset "Nonlinear solvers" include("nonlinear/nonlinear_solver.jl")
    @safetestset "Print statements" include("nonlinear/print_statements.jl")
end
if "slow" in GROUPS
    @safetestset "Doctests" include("quality/doctests.jl")
end
