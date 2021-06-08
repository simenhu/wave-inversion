export  general_one_dimensional_wave_equation_with_parameters, vector_excitation, wave_equation_system_matrix

using DiffEqOperators
using RecursiveArrayTools
using DifferentialEquations
using OrdinaryDiffEq
using Zygote: @ignore, Buffer
using Infiltrator
using Profile
using Plots
using Tracker
using SparseArrays
using ForwardDiff

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
function excitation_density(a, positions, function_array, t)
    
    excitation_array = spzeros(eltype(a), size(a)...)

    for i in eachindex(positions)
        excitation_array[positions[i]] = function_array[i](t)
    end

    return excitation_array
end



function general_one_dimensional_wave_equation_with_parameters(domain, internal_nodes; function_array, excitation_positions, pml_width)
    
    dx = domain/(internal_nodes+1)
    pml_coeffs = dampening_coefficients(internal_nodes, pml_width; max_value=2000.0, order=2)


    function du_func(state, p, t)
        len_state = length(state)
        u_indexes = 1:div(len_state, 2)
        v_indexes = div(len_state, 2)+1:len_state

        u = @view state[u_indexes] # Electric field
        v = @view state[v_indexes] # Magnetic field

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

        if !(eltype(state) <: ForwardDiff.Dual)
            @infiltrate 
        end

        return [du; dv]
    end

    return du_func
end

function wave_equation_system_matrix(domain, internal_nodes, p, order=2; full_model=false)
    
    pml_width = 60

    dx = domain/(internal_nodes+1)

    a_coeffs = p[:,1] # 1/μ (1/permeability)
    b_coeffs = p[:,2] # 1/ϵ (1/permittivity)

    squared_size = internal_nodes + 2

    pml_coeffs = dampening_coefficients(internal_nodes, pml_width; max_value=2000.0, order=2)

    A_xv_array = zeros(squared_size, squared_size)
    A_xu_array = zeros(squared_size, squared_size)

    A_xv = Array(LeftStaggeredDifference{1}(1, order, dx, internal_nodes, b_coeffs))
    A_xu = Array(RightStaggeredDifference{1}(1, order, dx, internal_nodes, a_coeffs))

    A_xv_array[2:end-1, :] = A_xv
    A_xu_array[2:end-1, :] = A_xu

    if full_model
        A_xv_array[2:end-1, 2:end-1] = A_xv_array[2:end-1, 2:end-1] - diagm(pml_coeffs)
        A_xu_array[2:end-1, 2:end-1] = A_xu_array[2:end-1, 2:end-1] - diagm(pml_coeffs)
    end
    
    system_matrix = [I*0.0 A_xu_array; A_xv_array I*0.0]


    return system_matrix, A_xv_array, A_xu_array
end
