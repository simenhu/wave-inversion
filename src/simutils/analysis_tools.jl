export energy_of_string

using DifferentialEquations
using DiffEqOperators
using DataInterpolations

"""
Calculates the energy for a 1D-string solution for each time step of the solution and interpolates the
solution to enable ploting at any time step withhin the time range 
"""
function energy_of_string(sol, string_length, internal_cells, T, μ)
    dx = string_length/(internal_cells + 1)
    A_x = CenteredDifference(2, 3, dx, internal_cells)
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
