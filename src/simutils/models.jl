export  general_one_dimensional_wave_equation_with_parameters, vector_excitation

using DiffEqOperators
using RecursiveArrayTools
using DifferentialEquations
using OrdinaryDiffEq
using Zygote: @ignore, Buffer
using Infiltrator
using Profile
using Plots

"""
Returns an array with length equal to number of cells, with a quadratic ramp function
of the dampning coefficient at the ends
"""
function dampening_coefficients(number_of_cells, dampning_width; max_value=1., order=2)
    coeffs = zeros(number_of_cells)
    padding = Array(max_value.*range(0., 1., length=dampning_width).^order)
    coeffs[1:dampning_width] = padding[end:-1:1]
    coeffs[end-dampning_width+1: end] = padding
    return coeffs
end

"""
Makes a vector with the current density for excitation of the system.
"""
function excitation_density(a, positions::AbstractArray{T}, function_array, t) where {T<:Integer}
    buf = Buffer(a, length(a))
    
    for i in eachindex(a)
        buf[i] = zero(eltype(a))
    end

    for i in eachindex(positions)
        buf[positions[i]] = function_array[i](t)
    end

    return copy(buf)
end



function general_one_dimensional_wave_equation_with_parameters(domain, internal_nodes; function_array, excitation_positions, pml_width)
    
    dx = domain/(internal_nodes+1)
    pml_coeffs = dampening_coefficients(internal_nodes, pml_width; max_value=2000.0, order=2)


    function du_func(state, p, t)
        u = @view state.x[1][:] # Electric field
        v = @view state.x[2][:] # Magnetic field

        a_coeffs = p[:,1] # 1/μ (1/permeability)
        b_coeffs = p[:,2] # 1/ϵ (1/permittivity)
    
        A_xv = LeftStaggeredDifference{1}(1, 2, dx, internal_nodes, b_coeffs)
        A_xu = RightStaggeredDifference{1}(1, 2, dx, internal_nodes, a_coeffs)
        Q_v = Dirichlet0BC(Float64)
        Q_u = Dirichlet0BC(Float64)

        J = excitation_density(u, excitation_positions, function_array, t)

        # first equation
        du = A_xv*(Q_v*v) - b_coeffs.*J - u.*pml_coeffs

        # second equation
        dv = A_xu*(Q_u*u) - v.*pml_coeffs

        return ArrayPartition(du, dv)
    end
    return du_func
end

