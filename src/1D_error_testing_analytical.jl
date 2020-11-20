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
c_squared = T/μ
sim_time = (0.0, 10.0)
L = 2*pi

number_of_spatial_cells = 100
dx = L/(number_of_spatial_cells+1)
internal_positions = range(0, L, length=(number_of_spatial_cells + 2))[2:end-1]

# Initial conditions
x_0 = CuArray(2*sin.(internal_positions)) # Adding 2 to make the function even at the transitions
dx_0 = CUDA.zeros(number_of_spatial_cells)

c = sqrt(c_squared)
an_u(x,t) = 2*cos(c*t)*sin(x)
an_du(x,t) = -2*c*sin(c*t)*sin(x)

## Defining diff.eq
A_x = CenteredDifference(2, 3, dx, number_of_spatial_cells)
Q = Dirichlet0BC(Float64)

function ddx(ddx, dx, x, p, t)
    ddx[:] = c_squared*(A_x*Q)*x
end

prob = SecondOrderODEProblem(ddx, dx_0, x_0, sim_time)

## Simulate
solver =  AutoTsit5(Rosenbrock23())
sol = solve(prob, solver)

## Analysing error
"""
du = plot(sol[1].x[1])
u = plot(sol[1].x[2])
display(plot(u, du, layout = (2,1), size=(1400,600)))
"""

## Define function for energy
function energy_of_string(sol, T, μ, dx)
    A_x = CenteredDifference(2, 3, dx, length(sol[1].x[1]))
    Q = Dirichlet0BC(Float64)
    
    energy_array = zeros(length(sol))
    for i in eachindex(sol)
        du = sol[i].x[1]
        u = sol[i].x[2]
        e = 1/2*T*dot(-(A_x*Q)*u,u) + 1/2*μ*dot(du, du)
        energy_array[i] = e
    end
    return LinearInterpolation(energy_array, sol.t)
end

energy = energy_of_string(sol, T, μ, dx)
root_squared_error(x1, x2) = vec(sum(sqrt.((x1-x2).^2), dims=1))

## plot image of state compared with analytical solution
function analysis_plot(sol, analytical, sim_time, time_res, internal_positions; solver_name)
    state_dim = size(sol)[1]÷2

    time_vector = sim_time[1]:time_res:sim_time[2]
    analytical_sol = [analytical(x,t) for x in internal_positions, t in time_vector]
    simulated_sol = sol(time_vector)[state_dim+1:end,:]
    energy_vector = energy.(time_vector)
    RS_error = root_squared_error(analytical_sol, simulated_sol)

    clims = [-3.0, 3.0]

    title_plot = plot(title = solver_name, grid = false, showaxis = false, bottom_margin = -50Plots.px)
    analytical_plot = heatmap(time_vector, 1:state_dim, analytical_sol, title = "analytical")
    state_plot = heatmap(time_vector, 1:state_dim, simulated_sol, title = "simulated")
    error_plot = heatmap(time_vector, 1:state_dim, analytical_sol - simulated_sol, title = "difference")
    energy_plot = plot(time_vector, energy_vector, title = "energy", yaxis=[0., 16000.])
    rs_error = plot(time_vector, RS_error, title = "rs-error", yaxis=[0., 250.])
    display(plot(title_plot, analytical_plot, state_plot, error_plot, energy_plot, rs_error, layout = (6, 1), size=(1400, 900), link = :x, plot_title=solver_name))
end

analysis_plot(sol, an_u, sim_time, 0.005, internal_positions, solver_name = " AutoTsit5(Rosenbrock23())")