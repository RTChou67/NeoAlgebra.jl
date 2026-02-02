# ChemAlgebra.jl

Math tools for quantum chemistry implemented in Julia. This package provides specialized algorithms for electronic structure calculations, focusing on performance and ease of use.

## Benchmarks

Current performance comparisons against standard Julia packages (Arpack.jl, NLsolve.jl, Optim.jl).

### Davidson Diagonalization
*Comparison vs Arpack.jl (eigs)*

| Matrix Type | Size (N) | ChemAlgebra (ms) | Arpack (ms) | Speedup | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Near-Diag | 1000 | 0.57 | 5.04 | **8.8x** | PASS |
| Near-Diag | 5000 | 3.15 | 54.03 | **17.1x** | PASS |

> The custom Davidson implementation significantly outperforms Arpack for diagonally dominant matrices typical in CI/CC calculations.

### DIIS Acceleration
*Comparison vs NLsolve.jl (anderson)*

| Size (N) | ChemAlgebra (ms) | NLsolve (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| 500 | 1725.96 | 11507.13 | **6.7x** |
| 1000 | 7665.67 | 13956.37 | **1.8x** |

### Geometry Optimization (BFGS)
*Comparison vs Optim.jl*

| Case | ChemAlgebra (ms) | Optim.jl (ms) | Speedup | Status |
| :--- | :--- | :--- | :--- | :--- |
| Rosenbrock (2D) | 0.0037 | 0.0133 | **3.6x** | PASS |
| LJ-13 (Scaled) | 0.2760 | 0.0092 | 0.03x | PASS |

## TODO

- [ ] **BFGS Optimization**: Investigate performance bottleneck in Lennard-Jones (LJ-13) optimization. Current implementation is significantly slower (~30x) than Optim.jl for this case.
- [ ] **DIIS Scalability**: Analyze performance drop-off at larger N (speedup decreases from 6.7x to 1.8x).
- [ ] **Documentation**: Add detailed usage examples for `Davidson`, `DIIS`, and `BFGS` modules.
- [ ] **Testing**: Expand test suite to cover edge cases and non-diagonal matrices.