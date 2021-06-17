using DiffEqSensitivity: ZygoteRules
using Zygote: zeros, length
##
push!(LOAD_PATH, "./src/simutils/")
using DiffEqOperators
using DifferentialEquations
using DiffEqFlux
using OrdinaryDiffEq
using Optim
using Flux
using DiffEqSensitivity
using Zygote
using Random

using LinearAlgebra
using Plots
using TimerOutputs
using BenchmarkTools
using FiniteDifferences
using Infiltrator

plotlyjs()

using DelimitedFiles, Plots
using Simutils


## Defining constants for string property
T = 100.0 # N
μ = 0.01 # Kg/m
sim_time = (0.0, 0.14)
string_length = 2*pi
dx = 0.01
number_of_cells = Int(div(string_length, dx))
dt = 0.0001
abstol = 1e-8
reltol = 1e-8
receiver_position = 100

## Making inversion data
# Defining constants for time property
Δt = 0.001
t_vector = sim_time[1]:Δt:sim_time[2]
frequency = 50

f_excitation = gaussian_excitation_function(100, 0.005, sim_time, 0.03, 0.017)
internal_positions = internal_node_positions(0, string_length, number_of_cells)

## Initial conditions
u_0 = make_initial_condition(number_of_cells)
coeff_background = make_material_coefficients(number_of_cells, [sqrt(T/μ)], [[1]])

a_coeffs = copy(coeff_background)
b_coeffs = make_material_coefficients(number_of_cells, [sqrt(T/μ), sqrt(T/μ)*1.5], [[1], [400]])
p =  hcat(a_coeffs, b_coeffs)

## Define ODE function
f = general_one_dimensional_wave_equation_with_parameters(string_length, number_of_cells, function_array=[f_excitation], excitation_positions=[100], pml_width=60)
prob = ODEProblem(f, u_0, sim_time, p=p)

## Simulate
to = TimerOutput()
solvers =  [Tsit5(), TRBDF2(), Rosenbrock23(), AutoTsit5(Rosenbrock23()), Midpoint(), Vern7(),
             KenCarp4(), ORK256(), ParsaniKetchesonDeconinck3S32(), SSPRK22()]
solver = solvers[8]

sol = @timeit to "simulation" solve(prob, solver, save_everystep=true, p=p, abstol=abstol, reltol=reltol, dt=dt)
# display(animate_solution(sol, a_coeffs, b_coeffs, sim_time, 0.001))


solution_time = sol.t
function predict(Θ)
    Array(solve(prob, solver, p=Θ, saveat=solution_time; sensealg=InterpolatingAdjoint(),  abstol=abstol, reltol=reltol, dt=dt))
end


# loss(Θ) = error_loss(predict, sol, Θ)
# loss(Θ) = state_sum_loss(predict, sol,Θ)
loss(Θ) = error_position_loss(predict, sol, Θ, 100)
# loss(Θ) = error_position_frequency_loss(predict, sol, Θ, receiver_position , 15)
# loss(Θ) = energy_flux_loss_function(predict, sol, Θ, 100)
# loss(Θ) = error_position_frequency_energy_loss(predict, sol, Θ, 100, 15)
loss_with_freq(Θ, upper_frequency) = error_position_frequency_loss(predict, sol, Θ, receiver_position, upper_frequency)



## Parameters

perturbation_amplitude = 1e-5
step_length = 1e6

## Define helper funcions
function perturbate_vector(vec, amplitude_range)
    perturbation = 2*amplitude_range*rand(size(vec)...) .- amplitude_range
    vec + perturbation, perturbation
end

parameter_distance(x1, x2) = sqrt(sum((x1-x2).^2))

## Calculate initial values
b_coeffs_perturbated, perturbation = perturbate_vector(b_coeffs, perturbation_amplitude)

p_perturbated = [a_coeffs b_coeffs_perturbated]
loss_perturbated, wrong_sol = loss(p_perturbated)
display("loss from perturbation: $(loss_perturbated)")

## Gradient descent loop
loss_development = []
parameter_distance_development = []

last_loss = Inf
current_coeff = copy(p_perturbated)
current_loss = loss(current_coeff)[1]
push!(loss_development, current_loss)

current_parameter_distance = parameter_distance(p, p_perturbated)
push!(parameter_distance_development, current_parameter_distance)

while current_loss < last_loss
    global current_coeff
    global current_loss

    last_loss = current_loss
    current_grad = Zygote.gradient(Θ -> loss(Θ)[1], current_coeff)[1]
    current_coeff = current_coeff - step_length.*current_grad

    # Calculate new distance in parameter space
    current_parameter_distance = parameter_distance(current_coeff, p)
    push!(parameter_distance_development, current_parameter_distance)

    current_loss = loss(current_coeff)[1]
    push!(loss_development, current_loss)

    display("loss: $(current_loss), loss difference: $(current_loss - last_loss), parameter distance: $(current_parameter_distance)")
end

## Calculate solution with optimized parameters
optimized_sol = loss(current_coeff)[2]


## Plot metrics of convergence test
p11 = plot(loss_development, label="loss development", ylims=(0, max(loss_development...)*1.1))
p12 = plot(parameter_distance_development, label="parameter distance development", ylims=(0, max(parameter_distance_development...)*1.1), color=:red)
p1 = plot(p11, p12, layout=(2, 1))
display(p1)

p21 = plot(p_perturbated - p, label="Initial parameter error")
p22 = plot(current_coeff - p, label="Optimized parameter error")
p23 = plot(current_coeff - p_perturbated, label="Change in parameters")
p2 = plot(p21, p22, p23, layout=(3, 1), size=(750, 750))
display(p2)

p31 = plot(wrong_sol[100, :] - sol[100, :], label="Initial time domain error")
p32 = plot!(optimized_sol[100, :] - sol[100, :], label="Optimized time domain error")
display(p31)