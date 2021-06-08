##
push!(LOAD_PATH, "./src/simutils/")
using LinearAlgebra
using Plots
plotlyjs()
using Simutils
using ForwardDiff
using Zygote
using FiniteDifferences


## Defining constants for string property
T = 100.0 # N
μ = 0.01 # Kg/m
sim_time = (0.0, 0.05)
string_length = 2*pi
dx = 0.01
number_of_cells = Int(div(string_length, dx))

## Making inversion data
# Defining constants for time property
Δt = 0.001
t_vector = sim_time[1]:Δt:sim_time[2]
frequency = 50

f_excitation = gaussian_excitation_function(100, 0.005, sim_time, 0.03, 0.017)
internal_positions = internal_node_positions(0, string_length, number_of_cells)

## Initial conditions
# initial_position = sin.((2*pi/string_length)*internal_positions)
u_0 = make_initial_condition(number_of_cells)
a_coeffs = b_coeffs = make_material_coefficients(number_of_cells, [sqrt(T/μ), 1.5*sqrt(T/μ), 0.5*sqrt(T/μ)], [[1], [300], [450]])
Θ_0 = hcat(a_coeffs, b_coeffs)


## Define ODE function
f = general_one_dimensional_wave_equation_with_parameters(string_length, number_of_cells, function_array=[f_excitation], excitation_positions=[100], pml_width=60)

## calculate gradients of sum of model
state_func(u) = sum(f(u, Θ_0, 0.01))
coefficient_func(Θ) = sum(f(u_0, Θ, 0.01))


state_grad_forward = ForwardDiff.gradient(state_func, u_0)
state_grad_zygote = Zygote.gradient(state_func, u_0)[1]
# state_grad_finite = grad(central_fdm(2, 1), state_func, u_0)[1]

p1 = plot(state_grad_forward, label="state - ForwardDiff")
plot!(state_grad_zygote, label="state - zygote")
# plot!(state_grad_finite, label="state - finite")
display(p1)

parameter_grad_forward = ForwardDiff.gradient(coefficient_func, Θ_0)
parameter_grad_zygote = Zygote.gradient(coefficient_func, Θ_0)[1]
parameter_grad_finite = grad(central_fdm(2, 1), coefficient_func, Θ_0)[1]

p2 = plot(parameter_grad_forward[:, 1], label="a-coeff - ForwardDiff")
plot!(parameter_grad_zygote[:, 1], label="a-coeff - zygote")
plot!(parameter_grad_finite[:, 1], label="a-coeff - FiniteDifferences")

p3 = plot(parameter_grad_forward[:, 2], label="b-coeff - ForwardDiff")
plot!(parameter_grad_zygote[:, 2], label="b-coeff - zygote")
plot!(parameter_grad_finite[:, 2], label="b-coeff - FiniteDifferences")

display(p2)
display(p3)
