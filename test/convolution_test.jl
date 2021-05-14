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

## Test with simpler function
number_of_cells_2 = 628
dx = 0.01
u = sin.(internal_node_positions(0, 2*pi, number_of_cells_2+2))[2:end-1]
coeffs = sin.(internal_node_positions(0, 4*pi, number_of_cells_2+2))[2:end-1]


## Calculate the result using the concretizised matrix version of A
Ax = CenteredDifference{1}(1, 4, dx, number_of_cells_2, coeffs)
Q = Dirichlet0BC(Float64)

Ax_array = Array(Ax)
u_array = Array(Q*u)
du_array = Ax_array*u_array


## Calculate the results using non-alocating DiffEqOperators

du = Ax*(Q*u)

display("Error of the correct and non-allocating version is: $(sum(abs2, du_array - du))")
plot(du_array, label="correct")
plot!(du, label="non-allocating version")
plot(du_array - du, label="Error")