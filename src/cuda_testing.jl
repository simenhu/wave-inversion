##
push!(LOAD_PATH, "./src/simutils/")
using DiffEqOperators
using DifferentialEquations
using OrdinaryDiffEq
using LinearAlgebra
using Plots
using DataInterpolations
using Colors
using CUDA

plotlyjs()

using Simutils


## Defining constants for string property
ω = 1.0 
k = 1.0
T = 100.0 # N
μ = 0.01 # Kg/m
#c = (ω/k)^2
c_squared = Float32(T/μ)
sim_time = (0.0, 10.0)
L = 2*pi

number_of_spatial_cells = 100
dx = L/(number_of_spatial_cells+1)
internal_positions = internal

"""
x_0 =2*sin.(internal_positions) # Adding 2 to make the function even at the transitions
dx_0 = zeros(number_of_spatial_cells)

"""
## Initial conditions, CUDA
x_0 = CuArray{Float32}(2*sin.(internal_positions)) # Adding 2 to make the function even at the transitions
dx_0 = CUDA.zeros(number_of_spatial_cells)


c = sqrt(c_squared)
an_u(x,t) = 2*cos(c*t)*sin(x)
an_du(x,t) = -2*c*sin(c*t)*sin(x)

## Defining diff.eq
A_x = CenteredDifference(2, 3, dx, number_of_spatial_cells)
Q = Dirichlet0BC(Float32)

function ddx(ddx, dx, x, p, t)
    ddx[:] = c_squared*(A_x*Q)*x
end

function ddx_inplace(ddx, dx, x, p, t)
    mul!(ddx, c_squared*(A_x*Q), x)
end

prob = SecondOrderODEProblem(ddx_inplace, dx_0, x_0, sim_time)

## Simulate
solvers =  [Tsit5(), TRBDF2(), Rosenbrock23(), AutoTsit5(Rosenbrock23()), Midpoint(), Vern7()]
solver = solvers[1]
sol = solve(prob, solver)