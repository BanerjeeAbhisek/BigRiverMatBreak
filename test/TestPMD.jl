# test/Test_pmd.jl
include(joinpath(@__DIR__, "..", "src", "BRMB.jl"))
using .BRMB
using LinearAlgebra, Statistics, Random
Random.seed!(123456)

n, p, r = 300, 40, 3
X = randn(n, r) * randn(r, p) .+ 0.1 .* randn(n, p)
println("X is $n × $p\n")

# TEST 1 — correctness: at c = √p there's no sparsity, so the first sparse
# loading must equal the ordinary top PC (paper eq. 2.10), up to sign.
m_sp  = BRMB.pmd(X; k = 1, c = sqrt(p))
v_sp  = m_sp.loadings[:, 1]
v_ord = svd(X .- mean(X, dims = 1)).V[:, 1]
println("TEST 1  |⟨v_ordinary, v_sparse⟩| = ",
        round(abs(dot(v_ord, v_sp)), digits = 6), "   (want ≈ 1.0)\n")

# TEST 2 — sparsity appears as c shrinks
println("TEST 2  smaller c ⇒ fewer nonzero loadings")
for c in (sqrt(p), 4.0, 2.0, 1.2)
    local m = BRMB.pmd(X; k = 1, c = c)
    println("  c = $(round(c, digits = 2))  →  ",
            count(!iszero, m.loadings[:, 1]), " / $p features used")
end

# TEST 3 — multiple components, shapes
m3 = BRMB.pmd(X; k = 3, c = 2.0)
println("\nTEST 3  k=3, c=2.0")
println("  loadings size     : ", size(m3.loadings))
println("  nonzeros / column : ", [count(!iszero, m3.loadings[:, j]) for j in 1:3])





mort = BRMB.pmd_orth(X; k = 3, c = 2.0)
println("loadings size : ", size(mort.loadings))
println("nonzeros/col  : ", [count(!iszero, mort.loadings[:, j]) for j in 1:3])

Xc = X .- mean(X, dims = 1)

# scores from each method, then their correlation matrices
function offdiag_corr(scores)
    C = cor(scores)                      # correlations between component scores
    n = size(C, 1)
    maximum(abs(C[i, j]) for i in 1:n for j in 1:n if i != j)   # worst off-diagonal
end

T_pmd  = Xc * m3.loadings                # plain deflation (from Test 3)
T_orth = Xc * mort.loadings              # orthogonal variant
println("\nORTHOGONALITY  (max |off-diagonal score correlation|, lower = more orthogonal)")
println("  pmd      : ", round(offdiag_corr(T_pmd),  digits = 4))
println("  pmd_orth : ", round(offdiag_corr(T_orth), digits = 4))

# the real check: recover the u's and confirm they're mutually orthogonal.
# (pmd_orth doesn't store U, so recompute scores T = Xc*V and orthogonality
#  shows up as near-zero off-diagonal dot products among the u directions)