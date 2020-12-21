export internal_node_positions, string_with_coefficients!, make_initial_condition, string_with_coefficients_and_PML!

using DiffEqOperators
using RecursiveArrayTools
using DifferentialEquations
using OrdinaryDiffEq


"""
returns a range object which contains the positions of the internal nodes in a range
"""
function internal_node_positions(start, stop, number_of_spatial_cells)
    return range(start, stop, length=(number_of_spatial_cells+2))[2:end-1]
end

"""
Returns function for double derivative of string
"""
function string_with_coefficients!(string_length, number_of_cells, coefficients, excitation_functions, positions)
    dx = string_length/(number_of_cells+1)

    A_x = CenteredDifference{1}(2, 3, dx, number_of_cells, coefficients)
    Q = Dirichlet0BC(Float64)

    function ddu(ddx, dx, x, p, t)
        for i in eachindex(positions)
            x[positions[i]] = x[positions[i]] + excitation_functions[i](t)
        end
        mul!(ddx, A_x*Q, x)
    end
    return ddu
end

"""
Returns an array with length equal to number of cells, with a quadratic ramp function
of the dampning coefficient at hte ends
"""
function dampening_coefficients(number_of_cells, dampning_width; max_value=1., order=2)
    coeffs = zeros(number_of_cells)
    padding = Array(max_value.*range(0., 1., length=dampning_width).^order)
    coeffs[1:dampning_width] = padding[end:-1:1]
    coeffs[end-dampning_width+1: end] = padding
    return coeffs
end

display(plot(dampening_coefficients(100, 20, order=3)))

"""
Returns function for single derivative of sting
"""
function string_with_coefficients_and_PML!(string_length, number_of_cells; coefficients, excitation_func, positions, pml_width=30)
    dx = string_length/(number_of_cells+1)

    pml_coeffs = dampening_coefficients(number_of_cells, pml_width; max_value=1000.0, order=1)
    # c^2 = a*b 
    # c^2 = T/μ
    a = sqrt.(coefficients)
    b = sqrt.(coefficients)

    A_xv = LeftStaggeredDifference{1}(1, 2, dx, number_of_cells, b)
    A_xu = RightStaggeredDifference{1}(1, 2, dx, number_of_cells, a)
    Q_v = Dirichlet0BC(Float64)
    Q_u = Dirichlet0BC(Float64)

    function du_func(du, u, p, t)

        # if t > 0.1
        #     x = 1
        # end

        for i in eachindex(positions)
            u.x[2][positions[i]] = u.x[2][positions[i]] + excitation_func[i](t) # add the excitation value in the correct state
        end

        # first equation
        mul!(du.x[1], A_xv, Q_v*u.x[2])
        du.x[1] .= du.x[1] - u.x[1].*pml_coeffs
        

        # second equation
        mul!(du.x[2], A_xu, Q_u*u.x[1])
        du.x[2] .= du.x[2] - u.x[2].*pml_coeffs
    end
    return du_func
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