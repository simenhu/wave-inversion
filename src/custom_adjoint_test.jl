push!(LOAD_PATH, "./src/simutils/")
using Zygote
using Simutils
import ChainRulesCore: rrule, DoesNotExist, NO_FIELDS
using DiffEqOperators
using Plots
using LinearAlgebra
using Infiltrator
using Profile
using FiniteDiff
using FiniteDifferences
using ForwardDiff
plotlyjs()

## Test with simpler function
number_of_cells_2 = 628
dx = 0.01
u_2 = sin.(internal_node_positions(0, 2*pi, number_of_cells_2))
# coeffs_2 = [1.0 for i in 1:number_of_cells_2]
coeffs_2 = u_2 = cos.(2 .* internal_node_positions(0, 2*pi, number_of_cells_2))

function du(x, coeffs)
    Ax = RightStaggeredDifference{1}(1, 4, dx, number_of_cells_2, coeffs)
    Q = Dirichlet0BC(Float64)
    # @infiltrate
    return Ax*(Q*x)
end


## The result from 
res = du(u_2, coeffs_2)
res2 = du(u_2, coeffs_2.*2)
plot(res)
display(plot!(res2))

## Test derivatives

# @profview global grad_coeff = Zygote.gradient(coeffs -> sum(du(u_2, coeffs)), coeffs_2)[1]
grad_coeff = Zygote.gradient(coeffs -> sum(du(u_2, coeffs)), coeffs_2)[1]
finite_grad_coeff = grad(central_fdm(5,1), coeffs -> sum(du(u_2, coeffs)), coeffs_2)[1]
# forward_coeff = ForwardDiff.gradient(coeffs -> sum(du(u_2, coeffs)), coeffs_2)

grad_state = Zygote.gradient(u -> sum(du(u, coeffs_2)), u_2)[1]
finite_grad_state = grad(central_fdm(5,1), u -> sum(du(u, coeffs_2)), u_2)[1]
forward_state = ForwardDiff.gradient(u -> sum(du(u, coeffs_2)), u_2)

## Plot finite gradient compared to AD gradient
p1 = plot(grad_coeff, label="coeff - zygote")
p2 = plot!(finite_grad_coeff, label="coeff - finite diff")
# p3 = plot!(forward_coeff, label="coeff - forward diff")

p4 = plot(grad_state, label="state - zygote")
p5 = plot!(finite_grad_state, label="state - finite difference")
p6 = plot!(forward_state, label="state - forward")
display(plot(p1, p6, layout=(2,1), size=(700, 700)))