function Davidson(A::AbstractMatrix, n_roots::Int; max_iter = 100, tol = 1e-6)
	dim = size(A, 1)
	max_subspace = min(dim, n_roots * 20)
	diag_A = Vector(diag(A))
	n_guess = n_roots
	V = zeros(Float64, dim, max_subspace)
	Sigma = zeros(Float64, dim, max_subspace)
	sorted_indices = sortperm(diag_A)[1:n_guess]
	for (k, idx) in enumerate(sorted_indices)
		V[idx, k] = 1.0
	end
	curr_dim = n_guess
	mul!(view(Sigma, :, 1:curr_dim), A, view(V, :, 1:curr_dim))
	H_sub = zeros(Float64, max_subspace, max_subspace)
	for iter in 1:max_iter
		V_view = view(V, :, 1:curr_dim)
		Sigma_view = view(Sigma, :, 1:curr_dim)
		H_sub_view = view(H_sub, 1:curr_dim, 1:curr_dim)
		mul!(H_sub_view, V_view', Sigma_view)
		evals_sub, evecs_sub = eigen(Symmetric(H_sub_view))
		idx_sort = sortperm(evals_sub)
		current_evals = evals_sub[idx_sort[1:n_roots]]
		coeffs = view(evecs_sub, :, idx_sort[1:n_roots])
		max_res_norm = 0.0
		NewVecs = zeros(Float64, dim, n_roots)
		converged_cnt = 0
		for k in 1:n_roots
			E = current_evals[k]
			c = coeffs[:, k]
			r_vec = Sigma_view * c - V_view * c * E
			res_norm = norm(r_vec)
			max_res_norm = max(max_res_norm, res_norm)
			if res_norm < tol
				converged_cnt += 1
				NewVecs[:, k] .= 0.0
				continue
			else
				for i in 1:dim
					diff = E - diag_A[i]
					if abs(diff) < 1e-4
						diff = 1e-4
					end
					NewVecs[i, k] = r_vec[i] / diff
				end
			end
		end

		if iter % 10 == 0
			@printf("Davidson Iter %2d: E=%.8f MaxRes=%.2e Dim=%d\n",
				iter, current_evals[1], max_res_norm, curr_dim)
		end

		if max_res_norm < tol || converged_cnt == n_roots
			FinalVecs = V_view * coeffs
			return current_evals, FinalVecs
		end

		if curr_dim + n_roots > max_subspace
			RitzVecs = V_view * coeffs
			RitzSigma = Sigma_view * coeffs
			V[:, 1:n_roots] .= RitzVecs
			Sigma[:, 1:n_roots] .= RitzSigma
			curr_dim = n_roots
			continue
		end
		old_dim = curr_dim
		for k in 1:n_roots
			vec = view(NewVecs, :, k)
			for j in 1:curr_dim
				u = view(V, :, j)
				overlap = dot(u, vec)
				vec .-= overlap .* u
			end
			nrm = norm(vec)
			if nrm > 1e-12
				vec ./= nrm
				curr_dim += 1
				V[:, curr_dim] .= vec
				mul!(view(Sigma, :, curr_dim), A, view(V, :, curr_dim))
			end
		end
		if curr_dim == old_dim
			println("Davidson subspace stagnation detected. Terminating early.")
			break
		end
	end
	H_sub_view = view(H_sub, 1:curr_dim, 1:curr_dim)
	evals_sub, evecs_sub = eigen(Symmetric(H_sub_view))
	idx_sort = sortperm(evals_sub)
	coeffs = evecs_sub[:, idx_sort[1:n_roots]]
	FinalVecs = view(V, :, 1:curr_dim) * coeffs
	return evals_sub[idx_sort[1:n_roots]], FinalVecs
end
