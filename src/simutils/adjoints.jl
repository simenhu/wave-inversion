import DiffEqOperators
import LinearAlgebra
import ChainRulesCore: frule, rrule, DoesNotExist, NO_FIELDS, @thunk, Composite
using SparseArrays
using Infiltrator

export rrule

using Infiltrator

"""
Rule for the product of an DerivativeOperator and an AbstractArray
"""
function rrule(::typeof(*), A::DerivativeOperator, M::AbstractArray)
    Ώ = A*M
    function mul_pullback(ΔΩ)
        ∂A = (ΔΩ*M')
        # Same calculation as @thunk(A'*ΔΩ)
        ∂M = (left_multiply_adjoint(A, ΔΩ))
        return (NO_FIELDS, ∂A, ∂M)
    end
    return Ώ, mul_pullback 
end

"""
Rule for initialization of a BoundaryPaddedVector. This is the identity function
mapped back to its initializing arguments.
"""

"""
function rrule(::Type{<:BoundaryPaddedVector}, l, r, u)
    v = BoundaryPaddedVector(l, r, u)
    function BoundaryPaddedVector_pullback(ΔΏ)
        return (NO_FIELDS, ΔΏ[1], ΔΏ[2:end-1], ΔΏ[end])
    end
    return v, BoundaryPaddedVector_pullback
end
"""

"""
Rule for multiplication with a boundary condition object. This is actually an
initialization of a BoundaryPaddedVector which is the original vector padded with
zeroes at each end to fulfill the Dirichlet boundary conditions.
"""
function rrule(::typeof(*), Q::RobinBC, u::AbstractArray)
    b = Q*u
    function mul_pullback(ΔΏ)
        ∂u = ΔΏ[2:end-1]
        return (NO_FIELDS, DoesNotExist(), ∂u)
    end
    return b, mul_pullback
end


"""
Adjoint function for a DerivativeOperator which is staggered to work in wave
simulations. The Ax operator can be seen as an operator Ax = Dx * Cx where Dx is an
Operator with the same stencils as Ax before scaled by the spatialy dependent
constants of Cx. Cx has the spatially dependent constants allong it's diagonal.
"""
function rrule(::Type{<:RightStaggeredDifference}, derivative_order, approximation_order, dx, len, coeff_func)
    A = RightStaggeredDifference{1}(derivative_order, approximation_order, dx, len, coeff_func)
    _D = RightStaggeredDifference{1}(derivative_order, approximation_order, dx, len, 1.0)
    
    """
    The DerivativeOperator matrix is equal to the DerivativeOperator with 1.0 as
    coefficients where each column is scaled with a coefficient from a coefficient
    array. To get the pullback for the coefficient whe have to divide the pullback
    with the elements in DerivatorOperator with 1.0 coefficients
    """
    function RightStaggeredDifference_pullback(ΔΩ)
        # ∂c = diag(ΔΩ*_A')
        ∂c = right_multiply_diag_transpose(ΔΩ, _D)
        return (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist(), DoesNotExist(), ∂c)
    end
    return A, RightStaggeredDifference_pullback
end

function rrule(::Type{<:LeftStaggeredDifference}, derivative_order, approximation_order, dx, len, coeff_func)
    A = LeftStaggeredDifference{1}(derivative_order, approximation_order, dx, len, coeff_func)
    _D = LeftStaggeredDifference{1}(derivative_order, approximation_order, dx, len, 1.0)
    
    """
    The DerivativeOperator matrix is equal to the DerivativeOperator with 1.0 as
    coefficients where each column is scaled with a coefficient from a coefficient
    array. To get the pullback for the coefficient whe have to divide the pullback
    with the elements in DerivatorOperator with 1.0 coefficients
    """
    function LeftStaggeredDifference_pullback(ΔΩ)
        # ∂c = diag(ΔΩ*_A')
        ∂c = right_multiply_diag_transpose(ΔΩ, _D)
        return (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist(), DoesNotExist(), ∂c)
    end
    return A, LeftStaggeredDifference_pullback
end

function rrule(::typeof(excitation_density), a, positions, function_array, t)
    
    """
    The DerivativeOperator matrix is equal to the DerivativeOperator with 1.0 as
    coefficients where each column is scaled with a coefficient from a coefficient
    array. To get the pullback for the coefficient whe have to divide the pullback
    with the elements in DerivatorOperator with 1.0 coefficients
    """
    exciation_array = excitation_density(a, positions, function_array, t)

    function excitation_density_pullback(ΔΩ)
        return (NO_FIELDS, DoesNotExist(), DoesNotExist(), DoesNotExist(), DoesNotExist())
    end
    return exciation_array, excitation_density_pullback
end
