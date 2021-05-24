push!(LOAD_PATH, "./src/simutils/")
using Zygote
using Simutils
import ChainRulesCore: rrule, DoesNotExist, NO_FIELDS
using DiffEqOperators
using Plots
using LinearAlgebra
using Infiltrator
using Profile
using FiniteDifferences
using ForwardDiff
plotlyjs()

## Construct a derivative operator object
number_of_cells_2 = 628
dx = 0.01
u = sin.(internal_node_positions(0, 2*pi, number_of_cells_2+2))[2:end-1]
# coeffs = sin.(internal_node_positions(0, 4*pi, number_of_cells_2+2))[2:end-1]
coeffs = ones(number_of_cells_2)

Ax = RightStaggeredDifference{1}(1, 4, dx, number_of_cells_2, coeffs)
Q = Dirichlet0BC(Float64)

##