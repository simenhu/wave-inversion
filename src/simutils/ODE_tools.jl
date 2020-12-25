export second_order_to_first_order_system, second_to_first_order_initial_conditions, make_material_coefficients, internal_node_positions, make_initial_condition

using LinearAlgebra

function second_order_to_first_order_system(A)
    if !(ndims(A)==2 && size(A)[1]==size(A)[2])
        throw(ArgumentError("Matrix not square"))
    end
    dim = size(A)[1]
    first_order_matrix = [zeros(dim, dim) I; A zeros(dim, dim)] # Getting errors here when trying to concatinate array with GhostDerivativeOperator
    return first_order_matrix
end

function second_to_first_order_initial_conditions(u_dot_0, u_0)
    u = [u_0; u_dot_0]
end

"""
Fills an array with each value in coefficient_list in the intervals given in position
array. if one array in positions only have one element, the array is filled with the
coefficient in coefficient array from the starting index to the end.
"""
function make_material_coefficients(internal_nodes, coefficients, positions)
    coefficient_list = zeros(internal_nodes)
    for i in eachindex(coefficients)
        if length(positions[i])==2
            coefficient_list[position_list[i][1]:positions[i][2]] .= coefficients[i]
        else
            coefficient_list[positions[i][1]:end] .= coefficients[i]
        end
    end
    return coefficient_list
end

"""
returns a range object which contains the positions of the internal nodes in a range
"""
function internal_node_positions(start, stop, number_of_spatial_cells)
    return range(start, stop, length=(number_of_spatial_cells+2))[2:end-1]
end

"""
Returns arraypartition intended to use for initial state
"""
function make_initial_condition(number_of_cells, initial_condition=nothing)
    u0 = zeros(number_of_cells) # This is the variable which is double derivative in the original equation
    if initial_condition != nothing
        u0 = initial_condition
    end
    v0 = zeros(number_of_cells)
    uv0 = ArrayPartition(u0,v0)
end