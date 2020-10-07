##
push!(LOAD_PATH, "./src/simutils/")
using DiffEqOperators
using DifferentialEquations
using OrdinaryDiffEq
using LinearAlgebra
using Plots
plotlyjs()

using Simutils

## Defining constants
ω = 1.0 
k = 1.0
T = 100.0 # N
μ = 0.01 # Kg/m
#c = (ω/k)^2
#c = T/μ
c = 10000.0

number_of_spatial_cells = 100
dx = (2*pi)/(number_of_spatial_cells+1)

# Initial conditions
u_0 = 2*sin.(range(0, 2*pi, length=(number_of_spatial_cells + 2)))[2:end-1] # Adding 2 to make the function even at the transitions
u_dot_0 = zeros(number_of_spatial_cells)

## Analytical solution
analytical_u_dd_0 = -2*((2*pi)/100)^2*sin.(range(0, 2*pi, length=number_of_spatial_cells))

## Defining diff.eq
A_x = CenteredDifference(2, 2, dx, number_of_spatial_cells)
display(dx)
display(Array(A_x))
Q = Dirichlet0BC(Float64)
u_dot_dot(du, u, p, t) = c*(A_x*Q)*u
prob = SecondOrderODEProblem(u_dot_dot, u_dot_0, u_0, (0.0, 0.1))

## Simulate
sol = solve(prob, Tsit5())

## Showing results
du = plot(sol[1:100, :])
u = plot(sol[101:end, :])
display(plot(u, du, layout = (2,1), size=(1400,600)))