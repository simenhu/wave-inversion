using Plots: size
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
# plotlyjs()
pyplot()


## Test with simpler function
number_of_cells_2 = 628
dx = 0.01
u_0 = sin.(internal_node_positions(0, 2*pi, number_of_cells_2+2))[2:end-1]
# coeffs_2 = Array(range(1., 4., length=number_of_cells_2))
coeffs_0 = sin.(internal_node_positions(0, 4*pi, number_of_cells_2+2))[2:end-1]

function du(x, coeffs)
    Ax = RightStaggeredDifference{1}(1, 4, dx, number_of_cells_2, coeffs)
    Q = Dirichlet0BC(Float64)
    return Ax*(Q*x)
end

## Analyticall gradients

"Calculate gradient of states given coeficcient vector"
function analytical_state_grad(coeffs)
    # Does this since we never want the derivative of the coefficient at the end or
    # the boundary conditions
    D = Array(RightStaggeredDifference{1}(1, 4, dx, number_of_cells_2, 1.0))[:, 2:end-1]
    rows, columns = size(D) 
    grad = zeros(columns)
    for n in 1:columns
        row_sum = 0.0
        for m in 1:rows
            row_sum += D[m, n]*coeffs[m]
        end
        grad[n] = row_sum
    end
    return grad
end

"Calculate gradient of coefficients given state vector"
function analytical_coeff_grad(state)
    # Does this since we never want the derivative of the coefficient at the end or
    # the boundary conditions
    D = Array(RightStaggeredDifference{1}(1, 4, dx, number_of_cells_2, 1.0))[:, 2:end-1]
    rows, columns = size(D) 
    grad = zeros(rows)
    for m in 1:rows
        col_sum = 0.0
        for n in 1:columns
            col_sum += D[m, n]*state[n]
        end
        grad[m] = col_sum
    end
    return grad
end

coeff_analytical = analytical_coeff_grad(u_0)
state_analytical = analytical_state_grad(coeffs_0)

## Gradient wrt. coefficients
coeff_func = coeffs -> sum(du(u_0, coeffs))

coeff_zygote = Zygote.gradient(coeff_func, coeffs_0)[1]
coeff_difference = grad(central_fdm(5,1), coeff_func, coeffs_0)[1]
coeff_forward = ForwardDiff.gradient(coeff_func, coeffs_0)

## Gradient wrt. state
state_func = u -> sum(du(u, coeffs_0))

state_zygote = Zygote.gradient(state_func, u_0)[1]
state_difference = grad(central_fdm(2, 1), state_func, u_0)[1]
state_forward = ForwardDiff.gradient(state_func, u_0)

## Plot gradients
p1 = plot(coeff_zygote, label="coeff - zygote")
plot!(coeff_difference, label="coeff - finite diff")
plot!(coeff_forward, label="coeff - forward diff")
plot!(coeff_analytical, label="coeff - analytical")
plot!(size=(700, 350))
savefig("figures/custom_adjoint_test/coefficient_test_gradients.eps")
display(plot(p1))

p2 = plot(state_zygote, label="state - zygote")
plot!(state_difference, label="state - finite difference")
plot!(state_forward, label="state - forward")
plot!(state_analytical, label="state - analytical")
plot!(size=(700, 350))
savefig("figures/custom_adjoint_test/state_test_gradients.eps")
display(plot(p2))

## Plot state
p3 = plot(u_0, label="state")
plot!(size=(700, 350))
display(p3)
savefig("figures/custom_adjoint_test/state_zero_test_gradients.eps")

## Plot coefficients
p4 = plot(coeffs_0, label="coefficients")
plot!(size=(700, 350))
display(p4)
savefig(p4, "figures/custom_adjoint_test/coeffs_zero_test_gradients.eps")