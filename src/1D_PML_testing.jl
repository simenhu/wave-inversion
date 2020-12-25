##
push!(LOAD_PATH, "./src/simutils/")
using DiffEqOperators
using DifferentialEquations
using OrdinaryDiffEq
using LinearAlgebra
using Plots
using DataInterpolations
using TimerOutputs

plotlyjs()

using Simutils

## Defining constants for string property
T = 100.0 # N
μ = 0.01 # Kg/m
sim_time = (0.0, 0.25)
string_length = 2*pi
dx = 0.001
number_of_cells = Int(div(string_length, dx))

# Defining constants for time property
Δt = 0.001
t_vector = sim_time[1]:Δt:sim_time[2]
frequency = 50

f_excitation = gaussian_excitation_function(100, 0.005, sim_time, 0.03, 0.017)
internal_positions = internal_node_positions(0, string_length, number_of_cells)

## Initial conditions
u_0 = make_initial_condition(number_of_cells)
a_coeffs = b_coeffs = make_material_coefficients(number_of_cells, [sqrt(T/μ), 0.5*sqrt(T/μ)], [[1], [350]])

## Define ODE function
f = general_one_dimensional_wave_equation(string_length, number_of_cells, a_coeffs, b_coeffs, excitation_func=[f_excitation], excitation_positions=[100], pml_width=15)
prob = ODEProblem(f, u_0, sim_time)

## Simulate
to = TimerOutput()
solvers =  [Tsit5(), TRBDF2(), Rosenbrock23(), AutoTsit5(Rosenbrock23()), Midpoint(), Vern7()]
solver = solvers[6]

sol = @timeit to "simulation" solve(prob, solver)

## plot image of state compared with analytical solution
energy = energy_of_string(sol, string_length, number_of_cells, a_coeffs, T)
root_squared_error(x1, x2) = vec(sum(sqrt.((x1-x2).^2), dims=1))
excitation_energy_plot(sol, energy, f_excitation, sim_time, 0.0005, solver_name = repr(solver))
display(to)

## make animation
time_resolution = 0.002
anim = @gif for t = sim_time[1]:time_resolution:sim_time[2]
    plot(internal_positions, sol(t).x[1], legend=false, ylims=(-3,3))
end

# gif(anim, "Wave_into_dampening_layers.gif" ,fps=30)
