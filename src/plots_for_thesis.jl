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
# initial_position = sin.((2*pi/string_length)*internal_positions)
u_0 = make_initial_condition(number_of_cells)
# a_coeffs = b_coeffs = make_material_coefficients(number_of_cells, [sqrt(T/μ), 1.5*sqrt(T/μ), 0.5*sqrt(T/μ)], [[1], [300], [450]])
a_coeffs =  make_material_coefficients(number_of_cells, [sqrt(T/μ)], [[1]])
b_coeffs =  make_material_coefficients(number_of_cells, [sqrt(T/μ)], [[1]])
Θ  = hcat(a_coeffs, b_coeffs)
# Θ = make_material_coefficients(number_of_cells, [sqrt(T/μ)], [[1]])

## Define ODE function
f = general_one_dimensional_wave_equation_with_parameters(string_length, number_of_cells, Θ, excitation_func=[f_excitation], excitation_positions=[100], pml_width=60)
prob = ODEProblem(f, u_0, sim_time, p=Θ)

## Simulate
to = TimerOutput()
solvers =  [Tsit5(), TRBDF2(), Rosenbrock23(), AutoTsit5(Rosenbrock23()), Midpoint(), Vern7(), KenCarp4()]
solver = solvers[1]

sol = @timeit to "simulation" solve(prob, solver, save_everystep=true, p=Θ)
# bench = @benchmark solve(prob, solver, save_everystep=false, p=Θ)
display(to)
heatmap(sol[:,:])
##
display(animate_solution(sol, a_coeffs, b_coeffs, sim_time, 0.001))

## Energy plot
energy = energy_of_coupled_wave_equations(sol, a_coeffs, b_coeffs)
display(plot(energy))
