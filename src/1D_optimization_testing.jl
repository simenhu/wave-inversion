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
using FiniteDifferences

plotlyjs()

using DelimitedFiles, Plots
using DiffEqSensitivity, Zygote, Flux, DiffEqFlux, Optim
using Simutils


## Defining constants for string property
T = 100.0 # N
μ = 0.01 # Kg/m
sim_time = (0.0, 0.07)
string_length = 2*pi
dx = 0.01
number_of_cells = Int(div(string_length, dx))
dt = 0.0001
abstol = 1e-8
reltol = 1e-8

## Making inversion data
# Defining constants for time property
Δt = 0.001
t_vector = sim_time[1]:Δt:sim_time[2]
frequency = 50

f_excitation = gaussian_excitation_function(100, 0.005, sim_time, 0.03, 0.017)
internal_positions = internal_node_positions(0, string_length, number_of_cells)

## Initial conditions
u_0 = make_initial_condition(number_of_cells)
a_coeffs = make_material_coefficients(number_of_cells, [sqrt(T/μ)], [[1]])
b_coeffs = copy(a_coeffs)
Θ =  hcat(a_coeffs, b_coeffs)


## Define ODE function
f = general_one_dimensional_wave_equation_with_parameters(string_length, number_of_cells, function_array=[f_excitation], excitation_positions=[310], pml_width=60)
prob = ODEProblem(f, u_0, sim_time, p=Θ)

## Simulate
to = TimerOutput()
solvers =  [Tsit5(), TRBDF2(), Rosenbrock23(), AutoTsit5(Rosenbrock23()), Midpoint(), Vern7(), KenCarp4(), ORK256(), ParsaniKetchesonDeconinck3S32()]
solver = solvers[8]

sol = @timeit to "simulation" solve(prob, solver, save_everystep=true, p=Θ, abstol=abstol, reltol=reltol, dt=dt)
# display(animate_solution(sol, a_coeffs, b_coeffs, sim_time, 0.001))

## Calculate gradients
a_coeffs_start = make_material_coefficients(number_of_cells, [sqrt(T/μ)*1.01], [[1]])
b_coeffs_start = copy(a_coeffs_start)
Θ_start =  hcat(a_coeffs_start, b_coeffs_start)

wrong_sol = solve(prob, solver, save_everystep=true, p=Θ_start, abstol=abstol, reltol=reltol, dt=dt)

solution_time = sol.t
function predict(Θ)
    Array(solve(prob, solver, p=Θ, saveat=solution_time; sensealg=InterpolatingAdjoint(),  abstol=abstol, reltol=reltol, dt=dt))
end

function error_loss(Θ)
    pred = predict(Θ)
    l = pred - sol
    return sum(abs2, l), pred
end

function state_sum_loss(Θ)
    pred = predict(Θ)
    return sum(pred)
end

loss(Θ) = error_loss(Θ)
# loss(Θ) = state_sum_loss(Θ)

## Test that hte loss function makes sense
loss_with_correct_param = loss(Θ)[1]
loss_with_wrong_param = loss(Θ_start)[1]

display("Loss with correct param: $(loss_with_correct_param)")
display("Loss with wrong param: $(loss_with_wrong_param)")

# Define values for callback funciton
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
# @profview global grad_coeff = @timeit to "gradient calculation" Zygote.gradient(Θ -> loss(Θ)[1], Θ_start)
grad_coeff_zygote = @timeit to "gradient - zygote" Zygote.gradient(Θ -> loss(Θ)[1], Θ_start)[1]
p1 = plot(grad_coeff_zygote[:,1], label="a_coeffs, zygote")
p2 = plot!(grad_coeff_zygote[:, 2], label="b_coeffs, zygote")
display(p2)
display(to)


## Test gradient with small perturbation, ∇f⋅δ ≈ f(x+δ) - f(x)

iterations = 5

error_sum = 0.0
error_vector = zeros(iterations)

for i in 1:iterations
    global error_vector
    global error_sum
    delta = rand(size(Θ_start)...).*1e-6
    grad_dot_delta = dot(grad_coeff_zygote, delta)
    finite_delta = loss(Θ_start .+ delta)[1] - loss(Θ_start)[1]
    error = grad_dot_delta - finite_delta
    error_sum += error
    error_vector[i] = error
end

mean_error = error_sum/iterations
display("Mean error of finite difference test after $(iterations) iterations: $(mean_error)")
display(plot(error_vector))

## Test model with finite FiniteDifference
grad_coeff_finite = @timeit to "finite - gradient" grad(central_fdm(2, 1), Θ -> loss(Θ)[1], Θ_start)[1]
p3 = plot(grad_coeff_finite[:,1], label="a_coeffs, finite")
p4 = plot!(grad_coeff_finite[:, 2], label="b_coeffs, finite")
display(p4)
display(to)
