
            # simple Armijo line search
            #
            # λ = DEFAULT_ARMIJO_λ₀
            #
            # # δx = λ b
            # simd_copy_scale!(λ, s.linear.b, s.δx)
            #
            # for lsiter in 1:DEFAULT_LINESEARCH_nmax
            #     # x₁ = x₀ + λ δx
            #     simd_wxpy!(s.x₁, s.δx, s.x₀)
            #
            #     try
            #         # y₁ = f(x₁)
            #         function_stages!(s.x₁, s.y₁, s.Fparams)
            #
            #         # y₁norm = ||y₁||₂
            #         y₁norm = vecnorm(s.y₁, 2)
            #
            #         # exit loop if ||f(x₁)||₂ < (1-λϵ) ||f(x₀)||₂
            #         if y₁norm < (one(T)-λ*DEFAULT_ARMIJO_ϵ)*y₀norm
            #             break
            #         end
            #     catch DomainError
            #         # in case the new function value results in some DomainError
            #         # (e.g., for functions f(x) containing sqrt's or log's),
            #         # decrease λ and retry
            #         println("WARNING: Quasi-Newton Solver encountered Domain Error.")
            #         # if lsiter == DEFAULT_LINESEARCH_nmax
            #         #     λ *= 0
            #         #     simd_scale!(s.δx, 0)
            #         # end
            #     end
            #
            #     λ *= DEFAULT_ARMIJO_σ₁
            #     simd_scale!(s.δx, DEFAULT_ARMIJO_σ₁)
            # end
