import DiffEqOperators, ChainRules
import LinearAlgebra
import ChainRulesCore: frule, rrule, DoesNotExist, NO_FIELDS, @thunk
using SparseArrays

export mutation_testing, rrule

function mutation_testing(x)
    y = ones(10)
    for i in eachindex(y)
        y[i] = y[i]*x
    end
    return y
end


function rrule(::typeof(mutation_testing), x)
    
    function mutation_testing_pullback(ΔΏ)
        return (NO_FIELDS, x.*ΔΏ)
    end
    return mutation_testing(x), mutation_testing_pullback 
end



function rrule(::typeof(*), A::DerivativeOperator, M::AbstractArray)
    Ώ = A*M
    function mul_pullback(ΔΏ)
        A_sparse = SparseMatrixCSC(A)
        ∂A = @thunk(ΔΏ*M')
        ∂M = @thunk(A_sparse'*ΔΏ)
        return (NO_FIELDS, ∂A, ∂M)
    end
    return Ώ, mul_pullback 
end


function rrule(::Type{BoundaryPaddedVector}, l, r, u)
    v = BoundaryPaddedVector(l, r, u)
    function BoundaryPaddedVector_pullback(ΔΏ)
    end
    return v, BoundaryPaddedVector_pullback
end

# A bit uncertain if this is necessary. Shouldn't just the derivative of
# multiplication with an DerivativeOperator be necessary, since the initialization is
# taken care of the Right and Left staggered rules. 

# function rrule(::Type{DerivativeOperator}, derivative_order, approximation_order, dx, len, stencil_length, stencil_coefs, boundary_stencil_length, boundary_point_count, low_boundary_coefs, high_boundary_coefs, coefficients, coeff_func)
#     deriv_operator = DerivativeOperator{T,N,false,T,typeof(stencil_coefs), typeof(low_boundary_coefs),typeof(coefficients), typeof(coeff_func)}(
#         derivative_order, approximation_order, dx, len, stencil_length, stencil_coefs, boundary_stencil_length, boundary_point_count, low_boundary_coefs, high_boundary_coefs, coefficients, coeff_func)
    
#     concretized_deriv_operator = Array(DerivativeOperator{T,N,false,T,typeof(stencil_coefs), typeof(low_boundary_coefs),typeof(coefficients), typeof(coeff_func)}(
#         derivative_order, approximation_order, dx, len, stencil_length, stencil_coefs, boundary_stencil_length, boundary_point_count, low_boundary_coefs, high_boundary_coefs, coefficients, 1.0)) # Should maybe change this to type dependent one(...)
#     function DerivativeOperator_pullback(ΔΏ)
#             ∂_coefficients = concretized_deriv_operator'*ΔΏ
#         return (NO_FIELDS, DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(), ∂_coefficients)
#     end
#     return deriv_operator, DerivativeOperator_pullback
# end

function rrule(::Type{<:DerivativeOperator}, derivative_order, approximation_order, dx, len, stencil_length, stencil_coefs, boundary_stencil_length, boundary_point_count, low_boundary_coefs, high_boundary_coefs, coefficients, coeff_func)
    # Operator to be returnes as the primal
    A = DerivativeOperator{T,N,false,T,typeof(stencil_coefs), typeof(low_boundary_coefs),typeof(coefficients), typeof(coeff_func)}(
        derivative_order, approximation_order, dx, len, stencil_length, stencil_coefs, boundary_stencil_length, boundary_point_count, low_boundary_coefs, high_boundary_coefs, coefficients, coeff_func)
    
    # Operator to be used in calculating the  
    _A = SparseMatrixCSC(DerivativeOperator{T,N,false,T,typeof(stencil_coefs), typeof(low_boundary_coefs),typeof(coefficients), typeof(coeff_func)}(
        derivative_order, approximation_order, dx, len, stencil_length, stencil_coefs, boundary_stencil_length, boundary_point_count, low_boundary_coefs, high_boundary_coefs, coefficients, 1.0)) # Should maybe change this to type dependent one(...)
    function DerivativeOperator_pullback(ΔΏ)
            # ∂_coefficients = concretized_deriv_operator'*ΔΏ
            ∂_coefficients = diag(_A\ΔΏ)
        return (NO_FIELDS, DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(),DoesNotExist(), ∂_coefficients)
    end
    return A, DerivativeOperator_pullback
end

function rrule(::Type{<:RightStaggeredDifference}, derivative_order, approximation_order, dx, len, coeff_func)
    A = RightStaggeredDifference{1}(derivative_order, approximation_order, dx, len, coeff_func)
    _A = SparseMatrixCSC(RightStaggeredDifference{1}(derivative_order, approximation_order, dx, len, 1.0))
    function RightStaggeredDifference_pullback(ΔΏ)
        # ∂c = _A'*ΔΏ
        ∂c = diag(_A\ΔΏ)
        return (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist(), DoesNotExist(), ∂c)
    end
    return A, RightStaggeredDifference_pullback
end

function rrule(::Type{<:LeftStaggeredDifference}, derivative_order, approximation_order, dx, len, coeff_func)
    A = LeftStaggeredDifference{1}(derivative_order, approximation_order, dx, len, coeff_func)
    _A = SparseMatrixCSC(LeftStaggeredDifference{1}(derivative_order, approximation_order, dx, len, 1.0))
    
    # The DerivativeOperator matrix is equal to the DerivativeOperator_matrix with
    # 1.0 as coeff_func times a matrix with the coeff_func values as it's diagonal.
    # Ώ = _A*C. To get the coeff_func array, C,  back we take the diag of the matrix
    # _A^(-1)*Ώ or ΔΏ\_A   

    function LeftStaggeredDifference_pullback(ΔΏ)
        # ∂c = _A'*ΔΏ
        ∂c = diag(_A\ΔΏ)
        return (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist(), DoesNotExist(), ∂c)
    end
    return A, LeftStaggeredDifference_pullback
end

