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
a_coeffs = b_coeffs = make_material_coefficients(number_of_cells, [sqrt(T/μ)], [[1]])
Θ = (a_coeffs, b_coeffs)

## Define ODE function
f = general_one_dimensional_wave_equation_with_parameters(string_length, number_of_cells, Θ, excitation_func=[f_excitation], excitation_positions=[100], pml_width=60)
prob = ODEProblem(f, u_0, sim_time, p=Θ)

## Simulate
to = TimerOutput()
solvers =  [Tsit5(), TRBDF2(), Rosenbrock23(), AutoTsit5(Rosenbrock23()), Midpoint(), Vern7(), KenCarp4()]
solver = solvers[6]

sol = @timeit to "simulation" solve(prob, solver, save_everystep=true, p=Θ)
# bench = @benchmark solve(prob, solver, save_everystep=false, p=Θ)
display(to)
heatmap(sol[:,:])
##
animate_solution(sol, a_coeffs, b_coeffs, sim_time, 0.001)

## Optimization part

solution_time = sol.t
Θ_start = [ones(number_of_cells), ones(number_of_cells)]

function predict(Θ)
    Array(solve(prob, solver, p=Θ, saveat=solution_time))
end

function loss(Θ)
    pred = predict(Θ)
    l = pred - sol
    return sum(abs2, l)
end


l = loss(Θ_start)
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

# res = DiffEqFlux.sciml_train(loss, Θ_start, ADAM(0.01), cb = cb, maxiters = 100, allow_f_increases=false)  # Let check gradient propagation
# ps = res.minimizer
# display(ps)

##

du01, dp1 = Zygote.gradient(loss, Θ)

##https://mitmath.github.io/18337/

##

predict_sol = predict(Θ_start)

## Test derivative of derivative DiffEqOperators

test_number_of_nodes = 100
test_dx = 0.1
test_coefs = ones(test_number_of_nodes)
test_coefs[5:end] .= 2.0

Q = Dirichlet0BC(Float64)
A_x = RightStaggeredDifference{1}(1, 4, dx, test_number_of_nodes, test_coefs)

function f_test(x, p)
    Q = Dirichlet0BC(Float64)
    A_x = RightStaggeredDifference{1}(1, 4, dx, test_number_of_nodes, p)
    return A_x*(Q*x)
end


x_0_test = sin.((2pi/test_number_of_nodes).*1:test_number_of_nodes)

Zygote.gradient(p -> sum(f_test(x_0_test, p)), test_coefs)

##
f_test_2(p) = sum(f_test(x_0_test, p))
f_test_2(coefs)

## 

Zygote.gradient(coeff -> sum(Array(RightStaggeredDifference{1}(1,4,dx,test_number_of_nodes, coeff))), test_coefs)

## Testing of mutating array 

function mutation_testing(x)
    y = ones(10)
    for i in eachindex(y)
        y[i] = y[i]*x
    end
        return y
end

function rrule(::typeof(mutation_testing), x)
    
    function mutation_testing_pullback(ΔΏ)
        return (NO_FIELDS, ΔΏ*x)
    end
    return mutation_testing(x), mutation_testing_pullback 
 end

deriv_function(x) = sum(mutation_testing(x))

x_deriv_test = 5.
Zygote.gradient(deriv_function, x_deriv_test)