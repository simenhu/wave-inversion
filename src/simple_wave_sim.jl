##
push!(LOAD_PATH, "./src/simutils/")
using DiffEqOperators
using DifferentialEquations
using OrdinaryDiffEq
using LinearAlgebra
using Plots
using DataInterpolations

plotlyjs()

using Simutils

## Defining constants for string property
ω = 1.0 
k = 1.0
T = 100.0 # N
μ = 0.01 # Kg/m
#c = (ω/k)^2
c = T/μ
d = 0.0

number_of_spatial_cells = 100
dx = (2*pi)/(number_of_spatial_cells+1)

# Initial conditions
#x_0 = 2*sin.(range(0, 2*pi, length=(number_of_spatial_cells + 2)))[2:end-1] # Adding 2 to make the function even at the transitions
x_0 = zeros(number_of_spatial_cells)
dx_0 = zeros(number_of_spatial_cells)

# Defining constants for time property
sim_time = (0.0, 1.1)
Δt = 0.001
t = sim_time[1]:Δt:sim_time[2]
frequency = 50
x = sin.(2*pi*frequency*t)

# Exitation function
f = LinearInterpolation(x, t)


## Analytical solution
analytical_ddx_0 = -2*((2*pi)/100)^2*sin.(range(0, 2*pi, length=number_of_spatial_cells))

## Defining diff.eq
A_x = CenteredDifference(2, 3, dx, number_of_spatial_cells)
display(dx)
display(Array(A_x))
Q = Dirichlet0BC(Float64)

function ddx(ddx, dx, x, p, t)
    x[50] = x[50] + f(t)
    ddx[:] = c*(A_x*Q)*x
    #ddx[50] = f(t)
end

function exitate_x(func, exitation_functions, coords)
    return (ddx, dx, x)
end

prob = SecondOrderODEProblem(ddx, dx_0, x_0, sim_time)

## Simulate
sol = solve(prob, TRBDF2())

## Showing results

du = plot(sol[1:100, :])
u = plot(sol[101:end, :])
display(plot(u, du, layout = (2,1), size=(1400,600)))
