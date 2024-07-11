var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = SimpleSolvers","category":"page"},{"location":"#SimpleSolvers","page":"Home","title":"SimpleSolvers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SimpleSolvers]","category":"page"},{"location":"#SimpleSolvers.BisectionState","page":"Home","title":"SimpleSolvers.BisectionState","text":"simple bisection line search\n\n\n\n\n\n","category":"type"},{"location":"#SimpleSolvers.OptimizationAlgorithm","page":"Home","title":"SimpleSolvers.OptimizationAlgorithm","text":"An OptimizationAlgorithm is a datastructe that is used to dispatch on different algorithms.\n\nIt needs to implement three important methods,\n\ninitialize!(alg::OptimizationAlgorithm, ::AbstractVector)\nupdate!(alg::OptimizationAlgorithm, ::AbstractVector)\nsolver_step!(::AbstractVector, alg::OptimizationAlgorithm)\n\nthat initialize and update the state of the algorithm and perform an actual optimization step.\n\nFurther the following convenience methods should be implemented,\n\nobjective(alg::OptimizationAlgorithm)\ngradient(alg::OptimizationAlgorithm)\nhessian(alg::OptimizationAlgorithm)\nlinesearch(alg::OptimizationAlgorithm)\n\nwhich return the objective to optimize, its gradient and (approximate) Hessian as well as the linesearch algorithm used in conjunction with the optimization algorithm if any.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleSolvers.Options","page":"Home","title":"SimpleSolvers.Options","text":"Configurable options with defaults (values 0 and NaN indicate unlimited):\n\nx_abstol::Real = -Inf,\nx_reltol::Real = 2eps(),\nf_abstol::Real = 1e-50,\nf_reltol::Real = 2eps(),\nf_mindec::Real = 1e-4,\ng_restol::Real = sqrt(eps()),\nx_abstol_break::Real = Inf,\nx_reltol_break::Real = Inf,\nf_abstol_break::Real = Inf,\nf_reltol_break::Real = Inf,\ng_restol_break::Real = Inf,\nf_calls_limit::Int = 0,\ng_calls_limit::Int = 0,\nh_calls_limit::Int = 0,\nallow_f_increases::Bool = true,\nmin_iterations::Int = 0,\nmax_iterations::Int = 1_000,\nwarn_iterations::Int = max_iterations,\nshow_trace::Bool = false,\nstore_trace::Bool = false,\nextended_trace::Bool = false,\nshow_every::Int = 1,\nverbosity::Int = 1\n\n\n\n\n\n","category":"type"},{"location":"#SimpleSolvers.QuadraticState","page":"Home","title":"SimpleSolvers.QuadraticState","text":"Quadratic Polynomial line search\n\n\n\n\n\n","category":"type"},{"location":"#GeometricBase.value-Tuple{MultivariateObjective, Any}","page":"Home","title":"GeometricBase.value","text":"Evaluates the objective value at x. Returns f(x), but does not store the value in obj.f\n\n\n\n\n\n","category":"method"},{"location":"#GeometricBase.value-Tuple{MultivariateObjective}","page":"Home","title":"GeometricBase.value","text":"Get the most recently evaluated objective value of obj.\n\n\n\n\n\n","category":"method"},{"location":"#GeometricBase.value-Tuple{UnivariateObjective, Any}","page":"Home","title":"GeometricBase.value","text":"Evaluates the objective value at x. Returns f(x), but does not store the value in obj.F\n\n\n\n\n\n","category":"method"},{"location":"#GeometricBase.value-Tuple{UnivariateObjective}","page":"Home","title":"GeometricBase.value","text":"Get the most recently evaluated objective value of obj.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.derivative!!-Tuple{UnivariateObjective, Any}","page":"Home","title":"SimpleSolvers.derivative!!","text":"Force (re-)evaluation of the derivative of the objective at x. Returns f'(x) and stores the derivative in obj.D\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.derivative!-Tuple{UnivariateObjective, Any}","page":"Home","title":"SimpleSolvers.derivative!","text":"Evaluates the derivative of the objective at x. Returns f'(x) and stores the derivative in obj.D\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.derivative-Tuple{UnivariateObjective, Any}","page":"Home","title":"SimpleSolvers.derivative","text":"Evaluates the derivative of the objective at x. Returns f'(x), but does not store the derivative in obj.D\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.derivative-Tuple{UnivariateObjective}","page":"Home","title":"SimpleSolvers.derivative","text":"Get the most recently evaluated derivative of the objective of obj.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.gradient!!-Tuple{MultivariateObjective, Any}","page":"Home","title":"SimpleSolvers.gradient!!","text":"Force (re-)evaluation of the gradient at x. Returns ∇f(x) and stores the value in obj.g.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.gradient!-Tuple{MultivariateObjective, Any}","page":"Home","title":"SimpleSolvers.gradient!","text":"Evaluates the gradient at x. Returns ∇f(x) and stores the value in obj.g.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.gradient-Tuple{MultivariateObjective, Any}","page":"Home","title":"SimpleSolvers.gradient","text":"Evaluates the gradient at x. This does not update obj.g or obj.x_g.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.gradient-Tuple{MultivariateObjective}","page":"Home","title":"SimpleSolvers.gradient","text":"Get the most recently evaluated gradient of obj.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.isaOptimizationAlgorithm-Tuple{Any}","page":"Home","title":"SimpleSolvers.isaOptimizationAlgorithm","text":"Verifies if an object implements the OptimizationAlgorithm interface.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.linesearch_objective-Tuple{Any, Any, SimpleSolvers.NewtonSolverCache}","page":"Home","title":"SimpleSolvers.linesearch_objective","text":"create univariate objective for linesearch algorithm\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.linesearch_objective-Union{Tuple{T}, Tuple{MultivariateObjective, SimpleSolvers.NewtonOptimizerCache{T, AT} where AT<:(AbstractArray{T})}} where T","page":"Home","title":"SimpleSolvers.linesearch_objective","text":"create univariate objective for linesearch algorithm\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.update!-Tuple{Optimizer, AbstractVector}","page":"Home","title":"SimpleSolvers.update!","text":"compute objective and gradient at new solution and update result\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.value!!-Tuple{MultivariateObjective, Any}","page":"Home","title":"SimpleSolvers.value!!","text":"Force (re-)evaluation of the objective at x. Returns f(x) and stores the value in obj.f\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.value!!-Tuple{UnivariateObjective, Any}","page":"Home","title":"SimpleSolvers.value!!","text":"Force (re-)evaluation of the objective value at x. Returns f(x) and stores the value in obj.F\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.value!-Tuple{MultivariateObjective, Any}","page":"Home","title":"SimpleSolvers.value!","text":"Evaluates the objective at x. Returns f(x) and stores the value in obj.f\n\n\n\n\n\n","category":"method"},{"location":"#SimpleSolvers.value!-Tuple{UnivariateObjective, Any}","page":"Home","title":"SimpleSolvers.value!","text":"Evaluates the objective value at x. Returns f(x) and stores the value in obj.F\n\n\n\n\n\n","category":"method"}]
}
