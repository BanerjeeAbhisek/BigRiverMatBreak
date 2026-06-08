



# Benchmarking my PCA with Jchemo PCA for svd method .....


using BenchmarkTools, Jchemo, Statistics
using LinearAlgebra, Random
Random.seed!(1234)

# load YOUR pca
include(joinpath(@__DIR__, "..", "src", "BRMB.jl"))
#include("src/BRMB.jl")

using .BRMB

# test matrix — pick a shape; this one is tall
n, p, k = 5000, 200, 15
X = randn(n, p)

#  our SVD path 
b_mine = @btime BRMB.pca($X; k = $k, method = :svd);
# Run 1: 37.356 ms (77 allocations: 24.62 MiB)
# Run 2: 37.436 ms (77 allocations: 24.62 MiB)
# Run 3: 37.451 ms (77 allocations: 24.62 MiB)
# Run 4: 38.253 ms (77 allocations: 24.62 MiB)
# Run 5: 37.438 ms (77 allocations: 24.62 MiB)


#  Jchemo's SVD path 
b_jchemo = @btime pcasvd($X; nlv = $k);
# Run 1: 37.755 ms (124 allocations: 17.76 MiB)
# Run 2: 38.371 ms (124 allocations: 17.76 MiB)
# Run 3: 38.209 ms (124 allocations: 17.76 MiB)
# Run 4: 37.988 ms (124 allocations: 17.76 MiB)
# Run 5:  38.036 ms (124 allocations: 17.76 MiB)




# Benchmarking PMD with .....R 's PMD implementation in the PMA package.
# This is exactly the architecture mismatch that's causing the issue. Your Julia is native Apple Silicon (arm64), but your R is an Intel (x86_64) build. RCall can't bridge them 
# because it relies on shared libraries that are not compatible across architectures. To fix this, you would need to install an Apple Silicon version of R and the PMA package, 
# and then ensure that RCall is configured to use that version of R. Alternatively, you could run your Julia code in an x86_64 environment (like Rosetta 2) to match the architecture of your R installation,
# but this is less efficient than using native Apple Silicon versions of both.

# Checking wrt PCA 
@btime BRMB.pmd($X; k = $k, c = 3.0, maxiter = 5);
# Run 1: 43.353 ms (31260 allocations: 164.86 MiB)
# Run 2: 43.623 ms (31260 allocations: 164.86 MiB)
# Run 3: 42.953 ms (31260 allocations: 164.86 MiB)
# Run 4: 44.169 ms (31260 allocations: 164.86 MiB)
# Run 5: 43.723 ms (31260 allocations: 164.86 MiB)



# Checking with R SPC with same X metrix
using BenchmarkTools, DelimitedFiles, Random, LinearAlgebra, Statistics
Random.seed!(1234)

n, p = 200, 30
X = randn(n, p)
writedlm("Xshared.csv", X, ',')
println("saved Xshared.csv")

c = sqrt(p)                       # √30 ≈ 5.48, max budget (no sparsity)

# correctness (one run)
m     = BRMB.pmd(X; k = 1, c = c)
v_pmd = m.loadings[:, 1]
v_ord = svd(X .- mean(X, dims = 1)).V[:, 1]
println("pmd vs ordinary PCA : ", round(abs(dot(v_pmd, v_ord)), digits = 6))
# JULIA: pmd vs ordinary PCA : 1.0
# R: pmd vs ordinary PCA : 0.999542 
println("first 5 of v_pmd    : ", round.(v_pmd[1:5], digits = 4))
#  JULIA: first 5 of v_pmd    : [-0.2522, -0.0305,  0.3382,  0.2709,  0.0251]
#  R: first 5 of v_pmd    :     [0.2547,   0.0415, -0.3333, -0.2807, -0.0224 ]

# benchmark
println("\n--- timing ---")
@btime BRMB.pmd($X; k = 1, c = $c);
# 127.417 μs (412 allocations: 317.17 KiB)
#Julia pmd: ~127 μs = 0.127 ms (median)
# R SPC: 5.45 ms (median)








n, p = 1000, 100

# ---- Case 1: pure random 1000 × 100 (the degenerate case) ----
X_rand = randn(n, p)
m      = BRMB.pmd(X_rand; k = 1, c = sqrt(p))          # no sparsity
v_pmd  = m.loadings[:, 1]
v_ord  = svd(X_rand .- mean(X_rand, dims = 1)).V[:, 1]
println("PURE RANDOM  pmd vs ordinary PCA : ", round(abs(dot(v_pmd, v_ord)), digits = 6))

# ---- Case 2: structured data (clear dominant component) ----
X_struct = randn(n, 3) * randn(3, p) .+ 0.1 .* randn(n, p)
m2       = BRMB.pmd(X_struct; k = 1, c = sqrt(p))
v_pmd2   = m2.loadings[:, 1]
v_ord2   = svd(X_struct .- mean(X_struct, dims = 1)).V[:, 1]
println("STRUCTURED   pmd vs ordinary PCA : ", round(abs(dot(v_pmd2, v_ord2)), digits = 6))


using LinearAlgebra, Statistics, Random
include("src/BRMB.jl"); using .BRMB
Random.seed!(1234)

n, p = 1000, 100
nshow = 5                                  # how many loadings to print

# helper: print pmd's loading next to ordinary PCA's, aligned for sign
function compare_loadings(label, X; k = 1, nshow = 5)
    m     = BRMB.pmd(X; k = k, c = sqrt(size(X, 2)))   # no sparsity
    v_pmd = m.loadings[:, 1]
    v_ord = svd(X .- mean(X, dims = 1)).V[:, 1]

    # flip pmd's sign to match ordinary PCA's, so they're directly comparable
    if dot(v_pmd, v_ord) < 0
        v_pmd = -v_pmd
    end

    println("\n", label)
    println("  alignment |⟨pmd, PCA⟩| : ", round(abs(dot(v_pmd, v_ord)), digits = 6))
    println("  pmd loadings  : ", round.(v_pmd[1:nshow], digits = 4))
    println("  PCA loadings  : ", round.(v_ord[1:nshow], digits = 4))
end

# ---- Case 1: pure random ----
X_rand = randn(n, p)
compare_loadings("PURE RANDOM", X_rand; nshow = nshow)

# ---- Case 2: structured ----
X_struct = randn(n, 3) * randn(3, p) .+ 0.1 .* randn(n, p)
compare_loadings("STRUCTURED", X_struct; nshow = nshow)

