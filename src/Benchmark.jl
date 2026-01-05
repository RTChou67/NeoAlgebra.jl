
export benchmark_davidson, benchmark_diis, benchmark_optimization, run_all_benchmarks

function benchmark_davidson()
	Random.seed!(1234)
	n_roots = 4

	function measure_time(f)
		return (@belapsed $f() samples=20) * 1000
	end

	println("\n" * "="^90)
	println("BENCHMARK: Davidson (Eigenvalue Solver)")
	println("="^90)
	@printf("%-12s | %-6s | %-10s | %-10s | %-8s | %s\n",
		"Matrix", "N", "ChemAlg", "Arpack", "Speedup", "Status")
	println("-"^90)

	scenarios = [
		("Near-Diag", [1000, 5000], N -> begin
		D = spdiagm(0 => sort(rand(N)) .* 50.0)
		R = sprand(N, N, max(0.001, 10.0/N))
		D + 0.01 * (R + R')
	end)
	]

	for (label, dims, mat_gen) in scenarios
		for N in dims
			H = mat_gen(N)

			evals_ref, _ = eigs(H, nev = n_roots, which = :SR, tol = 1e-8)
			sort!(evals_ref)

			evals_my, _ = Davidson(H, n_roots, tol = 1e-8)
			sort!(evals_my)

			diff = norm(evals_my - evals_ref)
			status = diff < 1e-6 ? "PASS" : "FAIL"

			x0 = zeros(N)
			x0[1] = 1.0

			t_my = measure_time(() -> Davidson(H, n_roots, tol = 1e-6))
			t_ar = measure_time(() -> eigs(H, nev = n_roots, which = :SR, v0 = x0, tol = 1e-6))
			speedup = t_ar / t_my

			@printf("%-12s | %-6d | %10.2f | %10.2f | %7.1fx | %s (Err=%.1e)\n",
				label, N, t_my, t_ar, speedup, status, diff)
		end
	end
	println("-"^90)
end

function benchmark_diis()
	Random.seed!(1234)
	dims = [500, 1000]
	hist = 10

	scf_step(F) = 0.95 .* F .+ 0.05 .* sin.(F)
	nlsolve_f!(s, x) = (s .= x .- scf_step(x))

	function measure_time(f)
		return (@belapsed $f() samples=20) * 1000
	end

	println("\n" * "="^90)
	println("BENCHMARK: DIIS (Acceleration)")
	println("="^90)
	@printf("%-6s | %-10s | %-10s | %-8s\n", "N", "ChemAlg", "NLsolve", "Speedup")
	println("-"^90)

	for N in dims
		F0 = rand(N, N)

		function run_chem(F_in)
			mgr = DIISManager{Matrix{Float64}}(hist)
			F = copy(F_in)
			for _ in 1:200
				err = scf_step(F) - F
				if norm(err) < 1e-6
					break
				end
				F = diis_update!(mgr, F, err)
			end
			F
		end

		t_my = measure_time(() -> run_chem(F0))
		t_nl = measure_time(() -> nlsolve(nlsolve_f!, F0, method = :anderson, m = hist, ftol = 1e-6))

		@printf("%-6d | %10.2f | %10.2f | %7.1fx\n", N, t_my, t_nl, t_nl / t_my)
	end
	println("-"^90)
end

function rosen_fg_2d!(G, x)
	val = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
	if G !== nothing
		G[1] = -2.0 * (1.0 - x[1]) - 400.0 * x[1] * (x[2] - x[1]^2)
		G[2] = 200.0 * (x[2] - x[1]^2)
	end
	return val
end

function lj_fg!(G, x)
	N = length(x) ÷ 3
	coords = reshape(x, 3, N)
	E = 0.0
	if G !== nothing
		fill!(G, 0.0)
	end
	@inbounds for i in 1:N
		for j in (i+1):N
			dx = coords[1, i] - coords[1, j]
			dy = coords[2, i] - coords[2, j]
			dz = coords[3, i] - coords[3, j]
			r2 = dx^2 + dy^2 + dz^2
			if r2 < 0.01
				r2 = 0.01
			end
			inv_r2 = 1.0 / r2
			inv_r6 = inv_r2^3
			inv_r12 = inv_r6^2
			E += 4.0 * (inv_r12 - inv_r6)
			if G !== nothing
				factor = 24.0 * (inv_r6 - 2.0*inv_r12) * inv_r2
				gx = factor * dx
				gy = factor * dy
				gz = factor * dz
				G[3*i-2] += gx
				G[3*i-1] += gy
				G[3*i] += gz
				G[3*j-2] -= gx
				G[3*j-1] -= gy
				G[3*j] -= gz
			end
		end
	end
	return E
end

function rosen_f_optim(x)
	return rosen_fg_2d!(nothing, x)
end

function rosen_g_optim!(G, x)
	rosen_fg_2d!(G, x)
	return nothing
end

function lj_f_optim(x)
	return lj_fg!(nothing, x)
end

function lj_g_optim!(G, x)
	lj_fg!(G, x)
	return nothing
end

function get_lj13_init_structure()
	phi = (1.0 + sqrt(5.0)) / 2.0
	scale = 1.12
	coords = [
		0.0 1.0 phi
		0.0 -1.0 phi
		0.0 1.0 -phi
		0.0 -1.0 -phi
		1.0 phi 0.0
		-1.0 phi 0.0
		1.0 -phi 0.0
		-1.0 -phi 0.0
		phi 0.0 1.0
		phi 0.0 -1.0
		-phi 0.0 1.0
		-phi 0.0 -1.0
		0.0 0.0 0.0
	]
	return vec(transpose(coords .* scale))
end

function benchmark_optimization()
	Random.seed!(1234)

	function measure_time(f)
		return (@belapsed $f() samples=50) * 1000
	end

	println("\n" * "="^90)
	println("BENCHMARK: BFGS (Geometry Optimization)")
	println("="^90)
	@printf("%-18s | %-10s | %-10s | %-8s | %s\n",
		"Case", "ChemAlg", "Optim.jl", "Speedup", "Status")
	println("-"^90)

	x0_rosen = [-1.2, 1.0]

	_, E_my, _ = bfgs_optimize(rosen_fg_2d!, copy(x0_rosen), tol = 1e-8)

	res_opt = Optim.optimize(rosen_f_optim, rosen_g_optim!, copy(x0_rosen), Optim.BFGS(), Optim.Options(g_tol = 1e-8))
	E_ref = Optim.minimum(res_opt)

	diff = abs(E_my - E_ref)
	status = (diff < 1e-6) ? "PASS" : "FAIL"

	t_my = measure_time(() -> bfgs_optimize(rosen_fg_2d!, copy(x0_rosen), tol = 1e-8))
	t_opt = measure_time(() -> Optim.optimize(rosen_f_optim, rosen_g_optim!, copy(x0_rosen), Optim.BFGS(), Optim.Options(g_tol = 1e-8)))

	@printf("%-18s | %10.4f | %10.4f | %7.1fx | %s (Err=%.1e)\n",
		"Rosenbrock(2D)", t_my, t_opt, t_opt/t_my, status, diff)

	x_equil = get_lj13_init_structure()
	x0_lj = x_equil .* 1.05

	_, E_my_lj, _ = bfgs_optimize(lj_fg!, copy(x0_lj), tol = 1e-4)

	res_opt_lj = Optim.optimize(lj_f_optim, lj_g_optim!, copy(x0_lj), Optim.BFGS(), Optim.Options(g_tol = 1e-4))
	E_ref_lj = Optim.minimum(res_opt_lj)

	diff_lj = abs(E_my_lj - E_ref_lj)
	status_lj = (diff_lj < 1e-3) ? "PASS" : "FAIL"

	t_my_lj = measure_time(() -> bfgs_optimize(lj_fg!, copy(x0_lj), tol = 1e-4))
	t_opt_lj = measure_time(() -> Optim.optimize(lj_f_optim, lj_g_optim!, copy(x0_lj), Optim.BFGS(), Optim.Options(g_tol = 1e-4)))

	@printf("%-18s | %10.4f | %10.4f | %7.1fx | %s (Err=%.1e)\n",
		"LJ-13(Scaled)", t_my_lj, t_opt_lj, t_opt_lj/t_my_lj, status_lj, diff_lj)

	println("-"^90)
end

function run_all_benchmarks()
	benchmark_davidson()
	benchmark_diis()
	benchmark_optimization()
end
