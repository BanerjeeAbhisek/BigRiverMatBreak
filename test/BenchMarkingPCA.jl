# Benchmarking my PCA with Jchemo PCA for svd method .....


using BenchmarkTools, Jchemo
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




