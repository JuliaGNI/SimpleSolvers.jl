# The logic of the jobs of Dependents.yml. `harness workflows` copies this file to
# .github/dependents.jl of each base that the profile key `dependents_bases` lists, beside
# .github/workflows/Dependents.yml — edit this file and re-run the installer, never a copy.
#
#   julia dependents.jl list <dependents.toml>
#   julia dependents.jl excluded <base> <dependent>
#   julia dependents.jl row <name> <excluded> <head setup> <head test> <main setup> <main test>
#
# `list` prints the [[dependent]] tables of <dependents.toml> as a JSON array of objects with
# `name` and `repository`, the matrix of the test job. A missing file exits 2 with a line that
# names `harness dependents --apply`, which writes it.
#
# `excluded` prints `excluded=true` when the `version` of <base>/Project.toml lies outside the
# [compat] entry of <dependent>/Project.toml for the base, read with Pkg.Versions.semver_spec, and
# `excluded=false` when it lies inside or the dependent has no such entry; then one line with the
# reason. `Pkg.develop` refuses a head outside the entry, so the job tests no such head.
#
# A file that is not TOML, an entry that is malformed and a [compat] entry that is no version
# specifier exit 2 with one line on stderr.
#
# `row` classifies a dependent from `excluded` and the outcomes of the setup and the test steps
# against the head of the pull request and against the default branch of the base; `skipped` or
# empty is a step that did not run:
#
#   excluded `true`: excluded
#   head setup and test `success`, main did not run: green
#   head setup `failure`, main setup `success`: regression
#   head test `failure`, main setup and test `success`: regression
#   head test `failure`, main setup `success`, main test `failure`: known red
#   head setup or head test `failure`, main setup `failure`: inconclusive, with the failed setups
#
# After a failed setup, its test step does not run. `row` prints the row and appends it, under a
# header, to the job summary. A regression and an inconclusive row exit 1, which fails the job;
# a combination that is none of these exits 2.
#
# `list` and `excluded` also append their result to $GITHUB_OUTPUT, and `row` to
# $GITHUB_STEP_SUMMARY, where the variable is set.

using Pkg: Pkg
using TOML

const USAGE = "usage: dependents.jl list <dependents.toml> | excluded <base> <dependent> | row <name> <excluded> <head setup> <head test> <main setup> <main test>"
# A repository name, and a GitHub `owner/name`: the values that a [[dependent]] table may hold.
const NAME = r"^[A-Za-z0-9._-]+$"
const REPOSITORY = r"^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$"

function fail(message)
    println(stderr, "dependents.jl: ", message)
    exit(2)
end

"""
    append(variable, text)

Append `text` to the file that the environment variable `variable` names, where it is set.
"""
function append(variable, text)
    path = get(ENV, variable, "")
    isempty(path) || open(io -> print(io, text), path, "a")
    return nothing
end

"""
    readtoml(path)

The TOML file at `path`. A file that is not TOML ends the script with status 2.
"""
function readtoml(path)
    table = TOML.tryparsefile(path)
    table isa TOML.ParserError &&
        fail("$path is not a TOML file: $(table.type) at line $(table.line), column $(table.column)")
    return table
end

function list(path)
    isfile(path) ||
        fail("$path is missing: run `harness dependents --apply` and commit the file in the base")
    entries = get(readtoml(path), "dependent", [])
    entries isa Vector || fail("$path: `dependent` is not an array of tables")
    items = map(entries) do entry
        name = entry isa Dict ? get(entry, "name", nothing) : nothing
        repository = entry isa Dict ? get(entry, "repository", nothing) : nothing
        (name isa String && occursin(NAME, name) && repository isa String &&
         occursin(REPOSITORY, repository)) ||
            fail("$path: $(repr(entry)) is not a table with a name and a GitHub owner/name")
        "{\"name\":\"$name\",\"repository\":\"$repository\"}"
    end
    json = "[" * join(items, ",") * "]"
    println(json)
    append("GITHUB_OUTPUT", "dependents=$json\n")
    return 0
end

function excluded(base, dependent)
    path = joinpath(dependent, "Project.toml")
    head = readtoml(joinpath(base, "Project.toml"))
    name, version = get(head, "name", nothing), get(head, "version", nothing)
    (name isa String && version isa String &&
     tryparse(VersionNumber, version) !== nothing) ||
        fail("$(joinpath(base, "Project.toml")) has no name or no valid version")
    compat = get(get(readtoml(path), "compat", Dict()), name, nothing)
    compat === nothing || compat isa String ||
        fail("the [compat] entry for $name in $path is not a string")
    # semver_spec raises an ErrorException on a string that is no version specifier.
    spec = try
        compat === nothing ? nothing : Pkg.Versions.semver_spec(compat)
    catch e
        e isa ErrorException || rethrow()
        fail("the [compat] entry $(repr(compat)) for $name in $path is no version specifier")
    end
    out = spec !== nothing && !(VersionNumber(version) in spec)
    println("excluded=$out")
    println(compat === nothing ? "the dependent has no [compat] entry for $name" :
            "$name $version against the [compat] entry $(repr(compat))")
    append("GITHUB_OUTPUT", "excluded=$out\n")
    return 0
end

# GitHub reports a step that did not run as `skipped`, or as empty where the step is never reached.
notrun(outcome) = outcome in ("", "skipped")

function row(name, isexcluded, headsetup, headtest, mainsetup, maintest)
    headfailed = (headsetup == "failure" && notrun(headtest)) ||
                 (headsetup == "success" && headtest == "failure")
    result = if isexcluded == "true"
        "excluded"
    elseif isexcluded != "false"
        nothing
    elseif headsetup == "success" && headtest == "success" && notrun(mainsetup) &&
           notrun(maintest)
        "green"
    elseif !headfailed
        nothing
    elseif headsetup == "failure" && mainsetup == "success"
        "regression"
    elseif mainsetup == "success" && maintest == "success"
        "regression"
    elseif mainsetup == "success" && maintest == "failure"
        "known red"
    elseif mainsetup == "failure" && notrun(maintest)
        against = headsetup == "failure" ? "the head and the default branch" :
                  "the default branch"
        "inconclusive: the setup failed against $against"
    else
        nothing
    end
    result === nothing &&
        fail("cannot classify $name: excluded $(repr(isexcluded)), head setup $(repr(headsetup)), " *
             "head test $(repr(headtest)), main setup $(repr(mainsetup)), main test $(repr(maintest))")
    line = "| $name | $result |"
    println(line)
    append("GITHUB_STEP_SUMMARY", "| dependent | result |\n|:--|:--|\n$line\n")
    return result == "regression" || startswith(result, "inconclusive") ? 1 : 0
end

function main(args)
    command = isempty(args) ? "" : args[1]
    command == "list" && length(args) == 2 && return list(args[2])
    command == "excluded" && length(args) == 3 && return excluded(args[2], args[3])
    command == "row" && length(args) == 7 && return row(args[2:7]...)
    return fail(USAGE)
end

exit(main(ARGS))
