##
push!(LOAD_PATH, "./src/simutils/")
using DiffEqOperators
using DifferentialEquations
using OrdinaryDiffEq
using LinearAlgebra
using Plots
using DataInterpolations
using TimerOutputs
using BenchmarkTools
import ChainRules

plotlyjs()

using DelimitedFiles,Plots
using DiffEqSensitivity, Zygote, Flux, DiffEqFlux, Optim

using Simutils


## Defining constants for string property
T = 100.0 # N
μ = 0.01 # Kg/m
sim_time = (0.0, 0.5)
string_length = 2*pi
dx = 0.01
number_of_cells = Int(div(string_length, dx)) - 1 # 

# Making inversion data

# Defining constants for time property
Δt = 0.001
t_vector = sim_time[1]:Δt:sim_time[2]
frequency = 50

# f_excitation = gaussian_excitation_function(100, 0.005, sim_time, 0.03, 0.017)
f_excitation(t) = 0.0
internal_positions = internal_node_positions(0, string_length, number_of_cells)

## Initial conditions
u_0 = make_initial_condition(number_of_cells, gausian_state(number_of_cells,
 div(number_of_cells, 10), 100))
# u_0 = make_initial_condition(number_of_cells)
# a_coeffs = b_coeffs = make_material_coefficients(number_of_cells, [sqrt(T/μ), 1.5*sqrt(T/μ), 0.5*sqrt(T/μ)], [[1], [300], [450]])
a_coeffs =  make_material_coefficients(number_of_cells, [sqrt(T/μ)], [[1]])
b_coeffs =  make_material_coefficients(number_of_cells, [sqrt(T/μ)], [[1]])
Θ  = hcat(a_coeffs, b_coeffs)
# Θ = make_material_coefficients(number_of_cells, [sqrt(T/μ)], [[1]])

## Define ODE function
# f = general_one_dimensional_wave_equation_with_parameters(string_length, 
#     number_of_cells, Θ, excitation_func=[f_excitation], excitation_positions=[50],
#     pml_width=0)

f = general_one_dimensional_wave_equation_with_parameters(string_length, 
    number_of_cells, Θ, excitation_func=[f_excitation], excitation_positions=[50], pml_width=0)

    prob = ODEProblem(f, u_0, sim_time, p=Θ)

## Simulate
to = TimerOutput()
solvers =  [Tsit5(), TRBDF2(), Rosenbrock23(), AutoTsit5(Rosenbrock23()), Midpoint(), Vern7(), KenCarp4()]
solver = solvers[1]

sol = @timeit to "simulation" solve(prob, solver, save_everystep=true, p=Θ)
# bench = @benchmark solve(prob, solver, save_everystep=false, p=Θ)
display(to)


##  
# display(animate_solution(sol, a_coeffs, b_coeffs, sim_time, 0.001))

## Plot excitation signal
x_position = internal_positions
p = plot(internal_positions, u_0.x[1], xlabel="m", ylabel="m", label="Initial state", size=(700, 400))
times = [0.01, 0.02, 0.05, 0.1]
for t in times 
    plot!(internal_positions, sol(t).x[1], label= "t = "*string(t)*"s")
end

display(p)

## Energy plot
energy = energy_of_coupled_wave_equations(sol, a_coeffs, b_coeffs)

## Energy conservaton over time
time_res = 0.001
# display(excitation_energy_plot(sol, energy, f_excitation, sim_time, dx, time_res))
display(energy_plot(sol, energy, sim_time, dx, time_res))
