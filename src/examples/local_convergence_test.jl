using Optim: gradient_convergence_assessment
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

using DelimitedFiles
using Simutils

# plotlyjs()
pyplot()

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
perturbation_list = [1e-5, 1e-5, 1e-5] 
step_length_list = [1e6, 1e4, 1e2]
max_iter = 40
param_length = length(step_length_list)

## Define helper funcions
function perturbate_vector(vec, amplitude_range)
    vec = copy(vec)
    perturbation = 2*amplitude_range*rand(size(vec)...) .- amplitude_range
    vec + perturbation, perturbation
end

parameter_distance(x1, x2) = sqrt(sum((x1-x2).^2))

function gradient_descent_loop(coeff_perturbated, coeff_true, step_length)

    current_coeff = copy(coeff_perturbated)
    iter = 0
    loss_development = []
    current_loss = loss(current_coeff)[1]
    push!(loss_development, current_loss)

    parameter_distance_development = []
    current_parameter_distance = parameter_distance(coeff_true, current_coeff)
    push!(parameter_distance_development, current_parameter_distance)

    last_loss = Inf
    
    while current_loss < last_loss && iter<=max_iter
        iter += 1
        current_grad = Zygote.gradient(Θ -> loss(Θ)[1], current_coeff)[1]
        current_coeff = current_coeff - step_length.*current_grad

        # Calculate new distance in parameter space
        current_parameter_distance = parameter_distance(current_coeff, p)
        push!(parameter_distance_development, current_parameter_distance)

        current_loss, last_loss = loss(current_coeff)[1], current_loss
        push!(loss_development, current_loss)

        display("loss: $(current_loss), loss difference: $(current_loss - last_loss), parameter distance: $(current_parameter_distance)")
    end
    return current_coeff, loss_development, parameter_distance_development
end


## Gradient descent loop
optimized_coeff_collection = []
loss_collection = []
parameter_distance_collection = []
optimized_sol_collection = []
p_perturbated_collection = []
wrong_sol_collection = []

# Calculate convergence for three sets of parameters
for param in 1:param_length
    # Calculate initial values
    b_coeffs_perturbated, perturbation = perturbate_vector(b_coeffs, perturbation_list[param])

    p_perturbated = [a_coeffs b_coeffs_perturbated]
    push!(p_perturbated_collection, p_perturbated)
    loss_perturbated, wrong_sol = loss(p_perturbated)
    push!(wrong_sol_collection, wrong_sol)
    display("loss from perturbation $(perturbation_list[param]): $(loss_perturbated)")

    step_length = step_length_list[param]

    optimized_coeff, loss_development, parameter_distance_development = gradient_descent_loop(p_perturbated, p, step_length)
    optimized_sol = loss(optimized_coeff)[2]
    
    push!(optimized_coeff_collection, optimized_coeff)
    push!(loss_collection, loss_development)
    push!(parameter_distance_collection, parameter_distance_development)
    push!(optimized_sol_collection, optimized_sol)
end


## Plot metrics of convergence test
p1 = plot(ylabel="loss", xlabel="Iteration", legend=:top)
for param in 1:param_length
    _ = plot!(loss_collection[param], label="Loss: a,b=$(perturbation_list[param]), step-length=$(step_length_list[param])",yticks=0:30:max(loss_collection[param]...)*1.1, ylims=(0, max(loss_collection[param]...)*1.1))
end

p2 = plot(ylabel="euclidian distance", xlabel="iteration", legend=:bottomright)
for param in 1:param_length
    _ = plot!(parameter_distance_collection[param], label="Parameter distance: a,b=$(perturbation_list[param]), step-length=$(step_length_list[param])", ylims=(0, max(parameter_distance_collection[param]...)*1.1))
end


p1 = plot(p1, p2, layout=(2, 1))
display(p1)
savefig("figures/region_of_convergence_test/loss_euclidian_distance.eps")

parameter_index = 1
p21 = plot(p_perturbated_collection[parameter_index] - p, label="Initial coefficient error", xlabel="position")
p22 = plot(optimized_coeff_collection[parameter_index] - p, label="Optimized coefficient error", xlabel="position")
p23 = plot(optimized_coeff_collection[parameter_index] - p_perturbated_collection[parameter_index], label="Change in coefficients", xlabel="position")
p2 = plot(p21, p22, p23, layout=(3, 1), size=(750, 750))
display(p2)
savefig("figures/region_of_convergence_test/change_in_coefficients.eps")

p31 = plot(wrong_sol_collection[parameter_index][100, :] - sol[100, :], label="Initial time domain error", size=(700, 350))
p32 = plot!(optimized_sol_collection[parameter_index][100, :] - sol[100, :], label="Optimized time domain error", xlabel="time [s]")
display(p31)
savefig("figures/region_of_convergence_test/time_domain_error.eps")

## Calculate reduced loss values
loss_percentage_1 = 1 - parameter_distance_collection[1][end]/parameter_distance_collection[1][1]
loss_percentage_2 = 1 - parameter_distance_collection[2][end]/parameter_distance_collection[2][1]
loss_percentage_3 = 1 - parameter_distance_collection[3][end]/parameter_distance_collection[3][1]

display("Loss reduction with gamma=$(step_length_list[1]): $(loss_percentage_1)")
display("Loss reduction with gamma=$(step_length_list[2]): $(loss_percentage_2)")
display("Loss reduction with gamma=$(step_length_list[3]): $(loss_percentage_3)")

