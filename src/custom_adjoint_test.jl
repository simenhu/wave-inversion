push!(LOAD_PATH, "./src/simutils/")
using Zygote
using Simutils
import ChainRulesCore: rrule, DoesNotExist, NO_FIELDS
using Random
using DiffEqOperators
using Plots
using LinearAlgebra
using Infiltrator
using Profile
using FiniteDiff
using FiniteDifferences

plotlyjs()
Random.seed!(1)
## Defining constants for string property
T = 100.0 # N
μ = 0.01 # Kg/m
sim_time = (0.0, 0.15)
string_length = 2*pi
dx = 0.01
number_of_cells = Int(div(string_length, dx))

# Making inversion data

# Defining constants for time property
Δt = 0.001
t_vector = sim_time[1]:Δt:sim_time[2]
frequency = 50

f_excitation = gaussian_excitation_function(100, 0.005, sim_time, 0.03, 0.017)
internal_positions = internal_node_positions(0, string_length, number_of_cells)

## Initial conditions
u_0 = make_initial_condition(number_of_cells)
a_coeffs = b_coeffs = make_material_coefficients(number_of_cells, [sqrt(T/μ)], [[1]])
Θ = (a_coeffs, b_coeffs)

## Test with simpler function

number_of_cells_2 = 628
u_2 = sin.(internal_node_positions(0, 2*pi, number_of_cells_2))
coeffs_2 = [1.0 for i in 1:number_of_cells_2]

function simple_adjont_test_function()
    function du(x, coeffs)
        Ax = RightStaggeredDifference{1}(1, 4, dx, number_of_cells_2, coeffs)
        Q = Dirichlet0BC(Float64)
        return Ax*(Q*x)
    end
end
f_2 = simple_adjont_test_function()

@profview global grad_coeff = Zygote.gradient(coeffs -> sum(f_2(u_2, coeffs)), coeffs_2)[1]
display(size(grad_coeff))

## The result from 
res = f_2(u_2, coeffs_2)
res2 = f_2(u_2, coeffs_2.*2)
plot(res)
display(plot!(res2))

## Test derivative of state

grad_state = Zygote.gradient(u -> sum(f_2(u, coeffs_2)), u_2)[1]
size(grad_state)

## Code to analyze it agains sumerical solution_time

finite_grad_coeff = grad(central_fdm(5,1), coeffs -> sum(f_2(u_2, coeffs)), coeffs_2)[1]
finite_grad_state = grad(central_fdm(5,1), u -> sum(f_2(u, coeffs_2)), u_2)[1]

## Plot finite gradient compared to AD gradient
p1 = plot(grad_coeff, label="zygote coefficients")
p2 = plot!(finite_grad_coeff, label="finite difference coefficients")
p3 = plot(grad_state, label="zygote state")
p4 = plot!(finite_grad_state, label="finite difference state")
display(plot(p2, p4, layout=(2,1), size=(700, 700)))