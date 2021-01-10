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
sim_time = (0.0, 0.01)
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
# Θ  = hcat(a_coeffs, b_coeffs)
Θ = make_material_coefficients(number_of_cells, [sqrt(T/μ)], [[1]])

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
# display(animate_solution(sol, a_coeffs, b_coeffs, sim_time, 0.001))

## Optimization part

solution_time = sol.t
# Θ_start = hcat(ones(number_of_cells), ones(number_of_cells))
Θ_start = ones(number_of_cells)

function predict(Θ)
    Array(solve(prob, solver, p=Θ, saveat=solution_time))
end

function loss(Θ)
    pred = predict(Θ)
    l = pred - sol
    return sum(abs2, l), pred
end


l, pred = loss(Θ_start)
display(size(pred))
display(size(sol))
display(size(solution_time))

LOSS = []
PRED = []
PARS = []

cb = function(Θ, l, pred)
    display(l)
    append!(PRED, [pred])
    append!(LOSS, l)
    append!(PARS, [θ])
    false
end

## test gradient of loss
@profview global grad_coeff = @timeit to "gradient" Zygote.gradient(Θ -> loss(Θ)[1], Θ_start) 
display(plot(grad[1]))
display(to)

## Optimization

# res = DiffEqFlux.sciml_train(loss, Θ_start, ADAM(0.01), cb = cb, maxiters = 100, allow_f_increases=false)  # Let check gradient propagation
# ps = res.minimizer
# display(ps)

## test predict function
predict(Θ_start)

## test loss function

loss(Θ_start)

## Test modell function
display(typeof(f(u_0, Θ_start, 0.0)))