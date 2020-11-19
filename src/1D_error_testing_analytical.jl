##
push!(LOAD_PATH, "./src/simutils/")
using DiffEqOperators
using DifferentialEquations
using OrdinaryDiffEq
using LinearAlgebra
using Plots
using DataInterpolations
using Colors

plotlyjs()

using Simutils

## Defining constants for string property
ω = 1.0 
k = 1.0
T = 100.0 # N
μ = 0.01 # Kg/m
#c = (ω/k)^2
c_squared = T/μ
sim_time = (0.0, 10.1)
L = 2*pi

number_of_spatial_cells = 100
dx = L/(number_of_spatial_cells+1)
internal_positions = range(0, L, length=(number_of_spatial_cells + 2))

# Initial conditions
x_0 = 2*sin.(internal_positions)[2:end-1] # Adding 2 to make the function even at the transitions
dx_0 = zeros(number_of_spatial_cells)

## Analytical solution

c = sqrt(c_squared)
an_u(x,t) = 2*cos(c*t)*sin(x)
an_du(x,t) = -2*c*sin(c*t)*sin(x)

## Defining diff.eq
A_x = CenteredDifference(2, 3, dx, number_of_spatial_cells)
display(dx)
display(Array(A_x))
Q = Dirichlet0BC(Float64)

function ddx(ddx, dx, x, p, t)
    ddx[:] = c_squared*(A_x*Q)*x
end

prob = SecondOrderODEProblem(ddx, dx_0, x_0, sim_time)

## Simulate
sol = solve(prob, TRBDF2())

## Analysing error
du = plot(sol[1:100, 1])
u = plot(sol[101:end, 1],)
display(plot(u, du, layout = (2,1), size=(1400,600)))

## plot image of state compared with analytical solution
time_res = 0.005
time_vector = sim_time[1]:time_res:sim_time[2]
analytical_sol = [an_u(x,t) for x in internal_positions[2:end-1], t in time_vector]
simulated_sol = sol(time_vector)[101:end,:]
analytical_plot = plot(Gray.(analytical_sol))
state_plot = plot(Gray.(simulated_sol))
error_plot = plot(Gray.(analytical_sol - simulated_sol))

display(plot(analytical_plot, state_plot, error_plot, layout = (3, 1), size=(1400, 600), link = :x))

## Define energy of a state
"""
want to integrate over the spatial region of the string
"""


"""
Function for calculating energy of a vector. Input should be state vector after simulating. This
means that the first part of vector is the state time derivative, and second part is the state. 
"""
## Define function for energy
function energy_of_1D_system(du, u, T, μ)
    A_x = CenteredDifference(2, 3, dx, number_of_spatial_cells)
    Q = Dirichlet0BC(Float64)
    energy = 1/2*T*dot(-(A_x*Q)*u,u) + 1/2*μ*dot(du, du)
end

##
errors = []
for i in 0:0.01:sim_time[2]
    e = energy_of_1D_system(sol(i).x[1], sol(i).x[2], T, μ)
    push!(errors, e)
end

plot(errors)
