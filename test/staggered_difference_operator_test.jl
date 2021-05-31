push!(LOAD_PATH, "./src/simutils/")

using Simutils
using Test
using Random
using ChainRulesTestUtils
using ChainRulesCore
using Infiltrator
using DiffEqOperators


## Parameters for padded operators
internal_nodes = 8
dx = 0.1 
coeffs = [1.0 for i in 1:internal_nodes]

u = [1.0 for i in 1:(internal_nodes)]

## Padded operator

Ax = StaggeredDifference{1}(1, 4, dx, internal_nodes, coeffs)
Q = Dirichlet0BC(Float64)

display(Array(Ax))
du = Ax*(Q*u)
display(du)
## 