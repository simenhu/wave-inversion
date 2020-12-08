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
sim_time = (0.0, 1.0)
string_length = 2*pi
number_of_spatial_cells = 100

# Defining constants for time property
Δt = 0.001
t_vector = sim_time[1]:Δt:sim_time[2]
frequency = 50
x_excitation = sin.(2*pi*frequency*t_vector)

f_excitation = gaussian_excitation_function(100, 0.005, sim_time, 0.1, 0.017)
internal_positions = internal_node_positions(0, string_length, number_of_spatial_cells)

## Initial conditions
x_0 = zeros(number_of_spatial_cells)
dx_0 = zeros(number_of_spatial_cells)
c_squared = zeros(100)
c_squared[1:70] .= T/μ
c_squared[70:end] .= 0.1*T/μ


f = in_place_1D_string_with_coefficients(string_length, number_of_spatial_cells, c_squared, [f_excitation], [50])
prob = SecondOrderODEProblem(f, dx_0, x_0, sim_time)

## timing
to = TimerOutput()

## Simulate
solvers =  [Tsit5(), TRBDF2(), Rosenbrock23(), AutoTsit5(Rosenbrock23()), Midpoint(), Vern7()]
solver = solvers[6]
sol = @timeit to "simulation" solve(prob, solver)

energy = energy_of_string(sol, string_length, number_of_spatial_cells, c_squared, T)
root_squared_error(x1, x2) = vec(sum(sqrt.((x1-x2).^2), dims=1))

## plot image of state compared with analytical solution
function excitation_plot(sol, excitation_func, sim_time, time_res; solver_name)
    state_dim = size(sol)[1]÷2

    time_vector = sim_time[1]:time_res:sim_time[2]
    excitation_wave = [excitation_func(t) for t in time_vector]
    simulated_sol = sol(time_vector)[state_dim+1:end,:]
    energy_vector = energy.(time_vector)

    clims = [-3.0, 3.0]

    title_plot = plot(title = solver_name, grid = false, showaxis = false, bottom_margin = -200Plots.px)
    excitation_plot = plot(time_vector, excitation_wave, title = "excitation signal")
    state_plot = heatmap(time_vector, 1:state_dim, simulated_sol, title = "simulated")
    energy_plot = plot(time_vector, energy_vector, title = "energy")
    display(plot(title_plot, excitation_plot, state_plot, energy_plot, layout = (4, 1), size=(1400, 900), link = :x, plot_title=solver_name))
end

excitation_plot(sol, f_excitation, sim_time, 0.0001, solver_name = repr(solver))
display(to)