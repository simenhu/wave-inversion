export  string_with_coefficients!, string_with_coefficients_and_PML!, general_one_dimensional_wave_equation, 
general_one_dimensional_wave_equation_with_parameters!, general_one_dimensional_wave_equation_with_parameters

using DiffEqOperators
using RecursiveArrayTools
using DifferentialEquations
using OrdinaryDiffEq
using Zygote: @ignore
using Infiltrator
using Profile

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


"""
General wave equation of two coupled first order differential equations with
dampening layer and genral a and b coefficients. Version where I spesialice for
changing parameters of derivative operators and take kare of prealocation of
cariables.
"""
function general_one_dimensional_wave_equation_with_parameters!(domain, internal_nodes, p; excitation_func, excitation_positions, pml_width)
    
    a_coeffs, b_coeffs = p
    
    dx = domain/(internal_nodes+1)
    pml_coeffs = dampening_coefficients(internal_nodes, pml_width; max_value=2000.0, order=2)
    
    A_xv = LeftStaggeredDifference{1}(1, 4, dx, internal_nodes, b_coeffs)
    A_xu = RightStaggeredDifference{1}(1, 4, dx, internal_nodes, a_coeffs)
    Q_v = Dirichlet0BC(Float64)
    Q_u = Dirichlet0BC(Float64)

    A_xvq_operator = A_xv*Q_v
    A_xuq_operator = A_xu*Q_u

    function du_func(du, u, p, t)

        a_coeffs, b_coeffs = p

        A_xv = LeftStaggeredDifference{1}(1, 4, dx, internal_nodes, b_coeffs)
        A_xu = RightStaggeredDifference{1}(1, 4, dx, internal_nodes, a_coeffs)
        Q_v = Dirichlet0BC(Float64)
        Q_u = Dirichlet0BC(Float64)

        # for i in eachindex(excitation_positions)
        #     u.x[1][excitation_positions[i]] = u.x[1][excitation_positions[i]] + excitation_func[i](t) # add the excitation value in the correct state
        # end

        # first equation
        A_xvq_operator = A_xv*Q_v
        mul!(du.x[1], A_xvq_operator, u.x[2])
        du.x[1] .= du.x[1] .- u.x[1].*pml_coeffs

        # second equation
        A_xuq_operator = A_xu*Q_u
        mul!(du.x[2], A_xuq_operator, u.x[1])
        du.x[2] .= du.x[2] .- u.x[2].*pml_coeffs
    end
    return du_func
end

function general_one_dimensional_wave_equation_with_parameters(domain, internal_nodes, p; excitation_func, excitation_positions, pml_width)
    
    # a_coeffs =  p[:,1]
    # b_coeffs =  p[:,2]
    
    dx = domain/(internal_nodes+1)
    pml_coeffs = dampening_coefficients(internal_nodes, pml_width; max_value=2000.0, order=2)

    function du_func(state, p, t)
        u = @view state.x[1][:]
        v = @view state.x[2][:]


        # a_coeffs = p[:,1]
        # b_coeffs = p[:,2]

        a_coeffs = p
        b_coeffs = p

        # @infiltrate
        
        A_xv = LeftStaggeredDifference{1}(1, 4, dx, internal_nodes, b_coeffs)
        A_xu = RightStaggeredDifference{1}(1, 4, dx, internal_nodes, a_coeffs)
        Q_v = Dirichlet0BC(Float64)
        Q_u = Dirichlet0BC(Float64)

        @ignore for i in eachindex(excitation_positions)
            u[excitation_positions[i]] = u[excitation_positions[i]] + excitation_func[i](t) # add the excitation value in the correct state
        end
    
        # first equation
        du = A_xv*(Q_v*v) - u.*pml_coeffs

        # second equation
        dv = A_xu*(Q_u*u) - v.*pml_coeffs

        return ArrayPartition(du, dv)
    end
    return du_func
end

function one_dimensional_wave_equation(domain, internal_nodes, p; excitation_func, excitation_positions, pml_width)
    
    dx = domain/(internal_nodes+1)
    pml_coeffs = dampening_coefficients(internal_nodes, pml_width; max_value=2000.0, order=2)

    function du_func(state, p, t)
        u = @view state.x[1][:]
        v = @view state.x[2][:]

        a_coeffs = p[:,1]
        b_coeffs = p[:,2]
        
        A_xv = LeftStaggeredDifference{1}(1, 4, dx, internal_nodes, b_coeffs)
        A_xu = RightStaggeredDifference{1}(1, 4, dx, internal_nodes, a_coeffs)
        Q_v = Dirichlet0BC(Float64)
        Q_u = Dirichlet0BC(Float64)

        @ignore for i in eachindex(excitation_positions)
            u[excitation_positions[i]] = u[excitation_positions[i]] + excitation_func[i](t) # add the excitation value in the correct state
        end
    
        # first equation
        du = A_xv*(Q_v*v) - u.*pml_coeffs

        # second equation
        dv = A_xu*(Q_u*u) - v.*pml_coeffs

        return ArrayPartition(du, dv)
    end
    return du_func
end