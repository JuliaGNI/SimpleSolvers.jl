# Detect whether a function *itself* emits log messages, as opposed to delegating them.
#
# Several of the package's reporting sites sit behind `@noinline` barriers taking nothing but
# numbers, a `Symbol` and an `Options`, because their natural callers are specialized on a merit
# closure or on a solver and would therefore re-infer and re-codegen the message code once per
# problem a solver is built for — see `report_linesearch_status`, which documents the 76 s this
# cost `GeometricIntegrators`' Runge-Kutta suite.
#
# `@warn` is expanded by the macro, so its `GlobalRef`s into `Base.CoreLogging` are already present
# in lowered code. That makes "this function does not log" checkable directly, on any Julia, without
# running anything, and independently of what the rest of the suite has already compiled. The
# alternative — counting `Base.specializations` after driving a few solvers — only ever samples the
# closures it happens to use, and needs a `before`/`after` delta because other testsets may have
# compiled the same code first.

refs_corelogging(e) = e isa GlobalRef ? e.mod === Base.CoreLogging :
                      e isa Expr ? any(refs_corelogging, e.args) : false

"""
    has_logging_code(f)

Whether any method of `f` contains a log message in its own body. `Base.code_lowered(f)` returns
one `CodeInfo` per method, so functions with several methods or with default arguments are covered
without naming their signatures.
"""
has_logging_code(f) = any(ci -> any(refs_corelogging, ci.code), Base.code_lowered(f))

"""
    logged(f)

The messages `f()` emits, as strings.

Used to pin a reporter's *verbosity gate* now that the reporters are separate functions: a wrong
gate in a future edit is otherwise silent. Preferred over `@test_logs` here because it asks whether
one specific message is present, and so is not upset by unrelated messages the same solve may emit
at the same verbosity.
"""
function logged(f)
    logger = Test.TestLogger()
    Base.CoreLogging.with_logger(f, logger)
    [string(r.message) for r in logger.logs]
end

logged_any(f, pattern) = any(m -> occursin(pattern, m), logged(f))
