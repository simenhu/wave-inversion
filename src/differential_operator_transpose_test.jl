using DiffEqOperators

## Test operator
dx = 0.1
internal_nodes = 9
coeffs = [1.0 for n in 1:9]
state_0 = [1.0 for n in 1:11]

Ax = RightStaggeredDifference{1}(1, 6, dx, internal_nodes, coeffs)
display(Array(Ax))

#=
This 
=#


## Test if transpose of concretization is equal to concretization of transpose

@which transpose(Ax)
transpose(Ax)
# Ax_concretization_transpose = transpose(Array(Ax))
# Ax_transpose = Array(transpose(Ax))

## Test if DiffernceOperators axept function with two dimensions
dx = 0.1
internal_nodes = 9
coeffs_2D = ones(internal_nodes)


Ax_2d = RightStaggeredDifference{3}(1, 6, dx, internal_nodes, coeffs_2D)
# Q_2d = Dirichlet0BC{Float64}
display(Array(Ax))
state = reshape([n for n in 1:9 for m in 1:11],11,9)

display(size(Ax_2d))

display(Ax*state[:,1])
