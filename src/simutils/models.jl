export  string_with_coefficients!, string_with_coefficients_and_PML!, general_one_dimensional_wave_equation, 
general_one_dimensional_wave_equation_with_parameters!, general_one_dimensional_wave_equation_with_parameters,
general_one_dimensional_wave_equation_with_parameters_non_staggered

using DiffEqOperators
using RecursiveArrayTools
using DifferentialEquations
using OrdinaryDiffEq
using Zygote: @ignore
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
Add values to the positions of the vector a. This function is made to enable making a
rrule from ChainRules.jl for the excitation of the wave simulation models.
"""
function vector_excitation(a, positions, function_array, t)
    for i in eachindex(positions)
        a[positions[i]] = a[positions[i]] + function_array[i](t)
    end
    a
end


function general_one_dimensional_wave_equation_with_parameters(domain, internal_nodes; function_array, excitation_positions, pml_width)
    
    dx = domain/(internal_nodes+1)
    pml_coeffs = dampening_coefficients(internal_nodes, pml_width; max_value=2000.0, order=2)


    function du_func(state, p, t)
        u = @view state.x[1][:]
        v = @view state.x[2][:]


        a_coeffs = p[:,1]
        b_coeffs = p[:,2]

        # a_coeffs = p
        # b_coeffs = p
    
        A_xv = LeftStaggeredDifference{1}(1, 2, dx, internal_nodes, b_coeffs)
        A_xu = RightStaggeredDifference{1}(1, 2, dx, internal_nodes, a_coeffs)
        Q_v = Dirichlet0BC(Float64)
        Q_u = Dirichlet0BC(Float64)

        u = vector_excitation(u, excitation_positions, function_array, t)

        # first equation
        du = A_xv*(Q_v*v) - u.*pml_coeffs

        # second equation
        dv = A_xu*(Q_u*u) - v.*pml_coeffs

        return ArrayPartition(du, dv)
    end
    return du_func
end

