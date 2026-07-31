# Assertions about what lowering already shows, without running anything: which functions emit log
# messages, and which box a captured variable.
#
# Both properties are settled by lowering rather than by optimisation, so both appear as `GlobalRef`s
# in `Base.code_lowered` and neither depends on how the session was started — unlike a byte count,
# which `--check-bounds=yes` changes completely (see `AS_A_CALLER_COMPILES_IT` below).

refs_global(e, mod::Module, name::Union{Symbol,Nothing}=nothing) =
    e isa GlobalRef ? (e.mod === mod && (isnothing(name) || e.name === name)) :
    e isa Expr ? any(a -> refs_global(a, mod, name), e.args) : false

# `Base.code_lowered(f)` returns one `CodeInfo` per method, so functions with several methods or with
# default arguments are covered without naming their signatures.
any_lowered(predicate, f) = any(ci -> any(predicate, ci.code), Base.code_lowered(f))

"""
    has_logging_code(f)

Whether any method of `f` contains a log message in its own body, as opposed to delegating it.

Several of the package's reporting sites sit behind `@noinline` barriers taking nothing but numbers,
a `Symbol` and an `Options`, because their natural callers are specialized on a merit closure or on a
solver and would otherwise re-infer and re-codegen the message code once per problem a solver is
built for — see `report_linesearch_status`. `@warn` is expanded by the macro, so its `GlobalRef`s
into `Base.CoreLogging` are already present in lowered code, which makes "this function does not log"
checkable for every reporter at once and independently of what the rest of the suite has compiled.
"""
has_logging_code(f) = any_lowered(e -> refs_global(e, Base.CoreLogging), f)

"""
    has_boxed_capture(f)

Whether any method of `f` boxes a local that a closure captures and someone mutates.

Lowering wraps such a local in a `Core.Box`, which costs an allocation and, worse, erases its type,
so whatever is built from it is inferred `Any` and allocates in turn. Two line searches counted their
merit evaluations that way, which is why a converged solve allocated at all — see `_bierlaire_fit`
and `wolfe_status`.
"""
has_boxed_capture(f) = any_lowered(e -> refs_global(e, Core, :Box), f)

"""
    logged_any(f, pattern)

Whether any message `f()` emits contains `pattern`.

Used to pin a reporter's *verbosity gate*, which is otherwise silent when a future edit gets it
wrong. Preferred over `@test_logs` here because it asks whether one specific message is present, and
so is not upset by unrelated messages the same solve may emit at the same verbosity.
"""
function logged_any(f, pattern)
    logger = Test.TestLogger()
    Base.CoreLogging.with_logger(f, logger)
    any(r -> occursin(pattern, string(r.message)), logger.logs)
end

"""
    AS_A_CALLER_COMPILES_IT

Whether this session compiles the package the way a caller does, so that a byte count measured in it
says something about the package rather than about the session.

`julia-actions/julia-runtest` passes `--check-bounds=yes`, which inhibits the inlining that keeps the
merit closures and ForwardDiff's `Dual` buffers off the heap: under it a converged solve allocates a
few hundred bytes for *every* line search, `Static` included, so no fixed number can be asserted.
Code coverage turns out not to matter, but it is excluded too, since it perturbs the same
optimisations. The structural assertions above hold either way; the byte counts guarded by this are
what a developer sees on a plain `Pkg.test()`.
"""
const AS_A_CALLER_COMPILES_IT = Base.JLOptions().check_bounds == 0 && Base.JLOptions().code_coverage == 0
