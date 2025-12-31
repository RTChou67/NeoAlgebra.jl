# src/Benchmark.jl

"""
	benchmark()

运行 ChemAlgebra (Davidson) 与 Julia 生态现代求解器的终极性能对比。
包含 "Near-Diag" (物理模拟) 和 "Random" (数学压力) 两种场景。
"""
function benchmark()
	Random.seed!(1234)
	n_roots = 4

	# 维度设置
	# Near-Diag: 模拟大体系 Hamiltonian，测试高达 20,000 维
	dims_neardiag = [1000, 5000, 10000, 20000]
	# Random: 稠密矩阵压力测试
	dims_random = [500, 1000, 2000]

	results = []

	println("="^110)
	println("🧪  CHEMALGEBRA BENCHMARK SUITE: The Battle for Eigenvalues  🧪")
	println("="^110)

	# ==========================================================================
	# 辅助函数：安全运行 Arpack (防止不收敛报错终止程序)
	# ==========================================================================
	function safe_arpack(matrix, n, guess_vec)
		try
			# 增加 maxiter 和 ncv 以提高收敛几率
			t = @belapsed eigs($matrix, nev = $n, which = :SR, tol = 1e-6,
				v0 = $guess_vec, maxiter = 5000, ncv = 20)
			return t * 1000
		catch
			return NaN
		end
	end

	# ==========================================================================
	# Scenario 1: Near-Diag (Sparse Diagonally Dominant)
	# 模拟量子化学 CI/DFT/GW 哈密顿量。这是 Davidson 的绝对主场。
	# ==========================================================================
	println("\n" * "-"^110)
	println("🔹 SCENARIO 1: Near-Diag Matrices (Simulating Physics Hamiltonians)")
	println("   Structure: Large Sparse, Diagonally Dominant. Good Separation.")
	println("-"^110)

	for N in dims_neardiag
		println("\n  >> Dimension N = $N ...")

		# 1. 构造矩阵
		diag_vals = sort(rand(N)) .* 50.0
		D = spdiagm(0 => diag_vals)
		R = sprand(N, N, max(0.001, 10.0/N))
		H_mat = D + 0.01 * (R + R')

		# 2. 构造公平的“智能初猜” (Smart Guess)
		#    所有算法都从对角元最小的那些轨道开始猜，公平竞争
		X0_block = zeros(Float64, N, n_roots)
		# 既然我们生成时已经 sort 了 diag_vals，前 n_roots 个就是最小的
		for i in 1:n_roots
			;
			X0_block[i, i] = 1.0;
		end
		x0_single = X0_block[:, 1] # 给只支持单向量的算法用

		times = Dict()

		# [1] ChemAlgebra (Davidson)
		print("     [1] ChemAlgebra (Yours)...... ")
		if N == dims_neardiag[1]
			;
			Davidson(H_mat, n_roots, max_iter = 2);
		end # Warmup
		t = @belapsed Davidson($H_mat, $n_roots, tol = 1e-6)
		times[:chem] = t * 1000
		@printf("Done. (%7.2f ms)\n", times[:chem])

		# [2] Arpack (Arnoldi)
		print("     [2] Arpack (Arnoldi)......... ")
		# 传入 v0 初猜
		t_arp = safe_arpack(H_mat, n_roots, x0_single)
		times[:arpack] = t_arp
		if isnan(t_arp)
			print("FAIL (No Convergence)\n")
		else
			@printf("Done. (%7.2f ms)\n", t_arp)
		end

		# [3] KrylovKit (Lanczos)
		print("     [3] KrylovKit (Lanczos)...... ")
		# 传入 x0 初猜
		t = @belapsed KrylovKit.eigsolve($H_mat, $x0_single, $n_roots, :SR, tol = 1e-6)
		times[:kk] = t * 1000
		@printf("Done. (%7.2f ms)\n", times[:kk])

		# [4] IterativeSolvers (LOBPCG)
		print("     [4] IterativeSolvers (LOBPCG) ")
		# LOBPCG 必须要有 Preconditioner 才能在稀疏矩阵上跑得快
		P = Diagonal(1.0 ./ diag(H_mat))
		t = @belapsed IterativeSolvers.lobpcg($H_mat, false, $X0_block, P = $P, tol = 1e-6)
		times[:lobpcg] = t * 1000
		@printf("Done. (%7.2f ms)\n", times[:lobpcg])

		push!(results, ("Near-Diag", N, times))
	end

	# ==========================================================================
	# Scenario 2: Random (Dense Random Symmetric)
	# 数学压力测试。没有对角占优特性，对角预处理基本失效。
	# ==========================================================================
	println("\n" * "-"^110)
	println("🔸 SCENARIO 2: Random Matrices (Stress Test)")
	println("   Structure: Dense, Symmetric, No Diagonal Dominance.")
	println("-"^110)

	for N in dims_random
		println("\n  >> Dimension N = $N ...")
		A = randn(N, N)
		H_mat = (A + A') / 2

		# 随机矩阵没有物理意义，用随机初猜即可
		X0_block = rand(N, n_roots)
		x0_single = X0_block[:, 1]

		times = Dict()

		print("     [1] ChemAlgebra (Davidson)... ")
		t = @belapsed Davidson($H_mat, $n_roots, tol = 1e-6)
		times[:chem] = t * 1000
		@printf("Done. (%7.2f ms)\n", times[:chem])

		print("     [2] Arpack (Arnoldi)......... ")
		t_arp = safe_arpack(H_mat, n_roots, x0_single)
		times[:arpack] = t_arp
		if isnan(t_arp)
			;
			print("FAIL\n");
		else
			;
			@printf("Done. (%7.2f ms)\n", t_arp);
		end

		print("     [3] KrylovKit (Lanczos)...... ")
		t = @belapsed KrylovKit.eigsolve($H_mat, $x0_single, $n_roots, :SR, tol = 1e-6)
		times[:kk] = t * 1000
		@printf("Done. (%7.2f ms)\n", times[:kk])

		print("     [4] IterativeSolvers (LOBPCG) ")
		# 随机稠密矩阵没有简单的 Preconditioner，只能裸奔
		t = @belapsed IterativeSolvers.lobpcg($H_mat, false, $X0_block, tol = 1e-6)
		times[:lobpcg] = t * 1000
		@printf("Done. (%7.2f ms)\n", times[:lobpcg])

		push!(results, ("Random", N, times))
	end

	# ==========================================================================
	# 3. 输出汇总
	# ==========================================================================
	println("\n" * "="^110)
	println("🏆 FINAL STANDINGS (Time in ms)")
	println("="^110)

	@printf("%-10s | %-6s | %-12s | %-12s | %-12s | %-12s | %-15s\n",
		"Type", "Dim", "ChemAlg", "Arpack", "KrylovKt", "LOBPCG", "Winner")
	println("-"^110)

	for (type, N, times) in results
		tc = times[:chem]
		ta = times[:arpack]
		tk = times[:kk]
		tl = times[:lobpcg]

		# 寻找最小值 (忽略 NaN)
		valid_times = filter(!isnan, [tc, ta, tk, tl])
		min_t = isempty(valid_times) ? Inf : minimum(valid_times)

		win_str = ""
		if min_t == tc
			;
			win_str = "ChemAlgebra 🚀";
		end
		if min_t == ta
			;
			win_str = "Arpack";
		end
		if min_t == tk
			;
			win_str = "KrylovKit";
		end
		if min_t == tl
			;
			win_str = "LOBPCG";
		end

		fmt(x) = isnan(x) ? "FAIL 💀" : @sprintf("%8.2f", x)

		@printf("%-10s | %-6d | %8s     | %8s     | %8s     | %8s     | %-15s\n",
			type, N, fmt(tc), fmt(ta), fmt(tk), fmt(tl), win_str)
	end
	println("-"^110)
end
