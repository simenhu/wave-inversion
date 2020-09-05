# Have to discretize the problem
#This script is to try to simulate the one dimensional wave equation
##
using DiffEqOperators
using DifferentialEquations
using Plots
plotlyjs()


ω = 1
k = 1
c = (ω/k)^2

number_of_spatial_cells = 100
dx = 1/(number_of_spatial_cells+1)
ord_time_deriv = 2
ord_spacial_deriv = 2
number_of_time_cells = 100
dt = 1/(number_of_time_cells + 1)

## Initial conditions
u_0 = sin.(range(0, 2*pi, length=number_of_spatial_cells))
u_dot_0 = zeros(number_of_spatial_cells)

A_x = CenteredDifference{1}(2, 2, dx, number_of_spatial_cells)
Q = Dirichlet0BC(Float64)

u_dot(du, u, p, t) = c*A_x*Q*u

## Simulate
prob = SecondOrderODEProblem(u_dot, u_dot_0, u_0, (0.0, 1.0))
sol = solve(prob, Tsit5())


##
plot(sol.u[9])


# Boundary conditions
# Initial state
# simulere
# Plotte
# Array med forskjellig karakterestikk
