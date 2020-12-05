export not_in_place_1D_string, in_place_1D_string, internal_node_positions, in_place_1D_string_with_coefficients

using DiffEqOperators


function not_in_place_1D_string(string_length, number_of_cells, T, μ, excitation_functions, positions)
    c_squared = T/μ
    dx = string_length/(number_of_cells+1)

    A_x = CenteredDifference(2, 3, dx, number_of_cells)
    Q = Dirichlet0BC(Float64)

    function ddx(ddx, dx, x, p, t)
        for i in eachindex(positions)
            x[positions[i]] = x[positions[i]] + excitation_functions[i](t)
        end
        ddx[:] = c_squared*(A_x*Q)*x 
    end
    return ddx
end

function in_place_1D_string(length, number_of_cells, T, μ, excitation_functions, positions)
    c_squared = T/μ
    dx = length/(number_of_cells+1)

    A_x = CenteredDifference(2, 3, dx, number_of_cells)
    Q = Dirichlet0BC(Float64)

    function ddx(ddx, dx, x, p, t)
        for i in eachindex(positions)
            x[positions[i]] = x[positions[i]] + excitation_functions[i](t)
        end
        mul!(ddx, c_squared*A_x*Q, x)
    end
    return ddx
end

function in_place_1D_string_with_coefficients(string_length, number_of_cells, coefficients, excitation_functions, positions)
    dx = string_length/(number_of_cells+1)

    A_x = CenteredDifference{1}(2, 3, dx, number_of_cells, coefficients)
    Q = Dirichlet0BC(Float64)

    function ddx(ddx, dx, x, p, t)
        for i in eachindex(positions)
            x[positions[i]] = x[positions[i]] + excitation_functions[i](t)
        end
        mul!(ddx, A_x*Q, x)
    end
    return ddx
end 

"""
returns a range object which contains the positions of the internal nodes in a range
"""
function internal_node_positions(start, stop, number_of_spatial_cells)
    return range(start, stop, length=(number_of_spatial_cells+2))[2:end-1]
end