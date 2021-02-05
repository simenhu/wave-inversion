push!(LOAD_PATH, "./src/simutils/")
import Base: *
using Zygote
using Plots
using FiniteDiff
using FiniteDifferences
plotlyjs()

struct DumbDerivativeOperator
    stencil_matrix::AbstractArray
    scaling_array::AbstractArray
end


A = DumbDerivativeOperator([5. 5.; 6. 6.],[5. 6.])

DumbDerivativeOperator(c) = DumbDerivativeOperator(ones(length(c), length(c)), c)

function *(A::DumbDerivativeOperator, u::AbstractArray)
    (A.stencil_matrix.*A.scaling_array')*u
end


function du(c, x)
    A = DumbDerivativeOperator(c)
    # display(A.stencil_matrix)
    # display(A.scaling_array)
    A*x
end

## Define dummy example
vector_size = 10

c0  = [Float64(i) for i in range(0, 1, length=vector_size)]
x0 = sin.([(pi/vector_size)*i for i in 1:vector_size])

coeff_func = c -> sum(du(c, x0))
state_func = x -> sum(du(c0, x))

display(coeff_func(c0))

## Calculate gradient w.r.t coefficient
coeff_grad_zygote = Zygote.gradient(coeff_func, c0)[1]
coeff_grad_differences = grad(central_fdm(5,1), coeff_func, c0)[1]

## Calculate gradient w.r.t state_dim
state_grad_zygote = Zygote.gradient(state_func, x0)[1]
state_grad_differences = grad(central_fdm(5,1), state_func, x0)[1]

p0 = plot(x0, label="state")
plot!(c0, label="coefficients")
p1 = plot(coeff_grad_zygote, label="Zygote coefficient gradients")
plot!(coeff_grad_differences, label="difference coefficient gradients")
p2 = plot(state_grad_zygote, label="Zygote state gradients")
plot!(state_grad_differences, label="difference state gradients")
plot(p0, p1, p2, layout=(3,1))
