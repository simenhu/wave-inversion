using Zygote
import ChainRulesCore: rrule, DoesNotExist, NO_FIELDS

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

deriv_function(x) = sum(mutation_testing(x))

x_deriv_test = 5.
Zygote.gradient(deriv_function, x_deriv_test)