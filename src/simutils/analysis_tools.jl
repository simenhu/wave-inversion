export energy_of_string, energy_of_coupled_wave_equations

using DifferentialEquations
using DiffEqOperators
using DataInterpolations

"""
Calculates the energy for a 1D-string solution for each time step of the solution and interpolates the
solution to enable ploting at any time step withhin the time range 
"""
function energy_of_string(sol, string_length, internal_cells, coeffs, T)
    dx = string_length/(internal_cells + 1)
    A_x = CenteredDifference(2, 3, dx, internal_cells)
    Q = Dirichlet0BC(Float64)
    μ = T ./ coeffs
    
    energy_array = zeros(length(sol))
    for i in eachindex(sol)
        du = sol[i].x[1]
        u = sol[i].x[2]
        e = 1/2*T*dot(-(A_x*Q)*u,u) + 1/2*dot(du, μ .*  du) # Elementwize multiplication to include spatial dependent mass density
        energy_array[i] = e
    end
    return LinearInterpolation(energy_array, sol.t)
end

function energy_of_coupled_wave_equations(sol, a, b)
    ɛ_r = 1 ./ b 
    μ_r = 2 ./ a

    energy_array = zeros(length(sol))
    for i in eachindex(sol)
        u = sol[i].x[1]
        v = sol[i].x[2]

        e = 1/2*dot(u, ɛ_r.*u) + 1/2*dot(v,μ_r.*v) # Elementwize multiplication to include spatial dependent mass density
        energy_array[i] = e
    end
    return LinearInterpolation(energy_array, sol.t)

end

