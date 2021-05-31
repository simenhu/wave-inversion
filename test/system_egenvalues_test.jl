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

## Initial conditions
u_0 = make_initial_condition(number_of_cells)
a_coeffs = make_material_coefficients(number_of_cells, [sqrt(T/μ)], [[1]])
b_coeffs = copy(a_coeffs)
Θ =  hcat(a_coeffs, b_coeffs)

## Calculate gradients
# a_coeffs_start = make_material_coefficients(number_of_cells, [sqrt(T/μ)*1.01], [[1]])
a_coeffs_start = make_material_coefficients(number_of_cells, [1.0], [[1]])
b_coeffs_start = copy(a_coeffs_start)
Θ_start =  hcat(a_coeffs_start, b_coeffs_start)

## Test stability properties with system

system_matrix, A_xv, A_xu = wave_equation_system_matrix(string_length, number_of_cells, Θ_start, 2, full_model=true)
eigen_values = eigen(system_matrix).values
display(plot(real.(eigen_values)))

## Plot the difference anti-hermitian property

system_adjoint_difference = system_matrix - (-system_matrix')
display(spy(system_adjoint_difference))
