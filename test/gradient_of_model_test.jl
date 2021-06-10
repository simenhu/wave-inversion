##
push!(LOAD_PATH, "./src/simutils/")
using LinearAlgebra
using Plots
using Simutils
using ForwardDiff
using Zygote
using FiniteDifferences
using PrettyTables

# plotlyjs()
pyplot()


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
# u_0 = make_initial_condition(number_of_cells)
u_0 = [sin.((2*pi/string_length)*internal_positions); sin.((2*pi/string_length)*internal_positions)]
a_coeffs = b_coeffs = make_material_coefficients(number_of_cells, [sqrt(T/μ), 1.5*sqrt(T/μ), 0.5*sqrt(T/μ)], [[1], [300], [450]])
Θ_0 = hcat(a_coeffs, b_coeffs)


## Define ODE function
f = general_one_dimensional_wave_equation_with_parameters(string_length, number_of_cells, function_array=[f_excitation], excitation_positions=[100], pml_width=60)

## calculate gradients of sum of model
state_func(u) = sum(f(u, Θ_0, 0.01))
coefficient_func(Θ) = sum(f(u_0, Θ, 0.01))


state_grad_forward = ForwardDiff.gradient(state_func, u_0)
state_grad_zygote = Zygote.gradient(state_func, u_0)[1]
state_grad_finite = grad(central_fdm(2, 1), state_func, u_0)[1]


## Make plot for state gradients
p11 = plot(state_grad_zygote, label="state - zygote")
# plot!(state_grad_forward, label="state - ForwardDiff")
p12 = plot(state_grad_finite, label="state - finite")
p1 = plot(p11, p12, layout=(2, 1), size=(700, 350))
savefig("figures/ODE_adjoint_test/state_grad_ODE_adjoint_test.eps")
display(p1)

parameter_grad_forward = ForwardDiff.gradient(coefficient_func, Θ_0)
parameter_grad_zygote = Zygote.gradient(coefficient_func, Θ_0)[1]
parameter_grad_finite = grad(central_fdm(2, 1), coefficient_func, Θ_0)[1]

## Make plot for coefficient gradients
p21 = plot(parameter_grad_forward[:, 1], label="a-coeff grad - ForwardDiff")
p22 = plot(parameter_grad_zygote[:, 1], label="a-coeff grad - zygote")
p23 = plot(parameter_grad_finite[:, 1], label="a-coeff grad - FiniteDifferences")
p2 = plot(p21, p22, p23, layout=(3 ,1), size=(700, 350))
savefig("figures/ODE_adjoint_test/a_coeff_grad_ODE_adjoint_test.eps")
display(p2)

p31 = plot(parameter_grad_forward[:, 2], label="b-coeff grad - ForwardDiff")
p32 = plot(parameter_grad_zygote[:, 2], label="b-coeff grad - zygote")
p33 = plot(parameter_grad_finite[:, 2], label="b-coeff grad - FiniteDifferences")
p3 = plot(p31, p32, p33, layout=(3, 1), size=(700, 350))
savefig("figures/ODE_adjoint_test/b_coeff_grad_ODE_adjoint_test.eps")
display(p3)

## Make plot for initial states

p4 = plot(u_0, size=(700, 350), label="state")
savefig("figures/ODE_adjoint_test/state_ODE_adjoint_test.eps")
display(p4)

p51 = plot(a_coeffs, size=(700, 350), label="a-coeff", color=:blue)
p52 = plot(b_coeffs, size=(700, 350), label="b-coeff", color=:red)
p5 = plot(p51, p52, layout=(2, 1), size=(700, 350))
savefig("figures/ODE_adjoint_test/coefficients_ODE_adjoint_test.eps")
display(p5)

##  Calculate mean absolute difference

mean_absolute_difference(x1, x2) = sum(abs.(x1 - x2))/length(x1)

state_finite_mad = mean_absolute_difference(state_grad_zygote, state_grad_finite)

a_coeff_forward_mad = mean_absolute_difference(parameter_grad_zygote[:, 1], parameter_grad_forward[:, 1])
a_coeff_finite_mad = mean_absolute_difference(parameter_grad_zygote[:, 1], parameter_grad_finite[:, 1])

b_coeff_forward_mad = mean_absolute_difference(parameter_grad_zygote[:, 2], parameter_grad_forward[:, 2])
b_coeff_finite_mad = mean_absolute_difference(parameter_grad_zygote[:, 2], parameter_grad_finite[:, 2])

numeric_data =  [state_finite_mad "nan";
                a_coeff_finite_mad a_coeff_forward_mad;
                b_coeff_finite_mad b_coeff_forward_mad]

rows = ["state", "a_coeff", "b_coeff"]

data = [rows numeric_data]

pretty_table(data, header = ["variable", "finite", "forward"])


