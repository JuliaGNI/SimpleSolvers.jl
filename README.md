# SimpleSolvers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGNI.github.io/SimpleSolvers.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaGNI.github.io/SimpleSolvers.jl/latest)
[![PkgEval Status](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/S/SimpleSolvers.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/S/SimpleSolvers.html)
[![Build Status](https://github.com/JuliaGNI/SimpleSolvers.jl/workflows/CI/badge.svg)](https://github.com/JuliaGNI/SimpleSolvers.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaGNI/SimpleSolvers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNI/SimpleSolvers.jl)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.4317189.svg)](https://doi.org/10.5281/zenodo.4317189)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This package provides simple linear and nonlinear solvers such as LU decomposition and Newton's method. Under a unified interface, it provides low-overhead implementations in pure Julia, applicable to a wide range of data types, and wraps methods from other Julia libraries. Nonlinear solvers can be used with linesearch algorithms. Jacobians can be computed via automatic differentiation, finite differences or manually.

## References

If you use SimpleSolvers.jl in your work, please consider citing it by

```
@misc{Kraus:2020:SimpleSolvers,
  title={SimpleSolvers.jl: Simple linear and nonlinear solvers in Julia},
  author={Kraus, Michael},
  year={2020},
  howpublished={\url{https://github.com/JuliaGNI/SimpleSolvers.jl}},
  doi={10.5281/zenodo.4317189}
}
```


## Development

### Git hooks

Two hooks live in `.githooks`. They are **not active in a fresh clone** — `core.hooksPath` is local
configuration and does not travel with a push — so enable them once per clone:

```sh
git config core.hooksPath .githooks
```

**`pre-commit`** acts on **staged `.jl` files only**, and exits immediately when a commit stages
none, so a documentation- or workflow-only commit is not slowed down by it:

- **JuliaFormatter `--check`**, honouring this repository's own `.JuliaFormatter.toml` — **blocks**
  the commit. Formatting is mechanical and always fixable.
- **`fatou lint`**, when `fatou` is installed — **advisory only**, and deliberately so: its
  `unused-import` rule does not follow `include`, so it flags the load-bearing imports of every
  module file.
- **`using <Package>`**, which catches a syntax error or a broken `include` — **blocks**.

**`pre-push`** runs the full test suite with `--check-bounds=auto`, but **only when pushing to
`main` or `master`**; a topic branch is left to CI. It prints nothing for **10–30 minutes**, which
looks exactly like a network hang and is not one. If you do interrupt it, check for an orphaned
Julia process that the killed hook left behind.

Either hook can be bypassed for a single command with `--no-verify`, for a change you know it does
not apply to:

```sh
git commit --no-verify
git push --no-verify
```

The hooks are generated from one shared copy and are byte-identical across the related
repositories, so edit them there rather than here — a local edit is silently undone by the next
install.
