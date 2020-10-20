export second_order_to_first_order_system, second_to_first_order_initial_conditions

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
