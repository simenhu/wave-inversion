## Importing
using DiffEqOperators

## Testing of LeftStaggeredDifference
number_of_nodes = 6
dx = 0.01
A_l = LeftStaggeredDifference{1}(1, 4, dx, number_of_nodes, [1.0 for i in 1:number_of_nodes])
display(Array(A_l))
A_r = RightStaggeredDifference{1}(1, 4, dx, number_of_nodes, [1.0 for i in 1:number_of_nodes])
display(Array(A_r))


## Test * operator with  boundary condition
display("left difference")
Q = Dirichlet0BC(Float64)
display(Array(A_l))
display(A_l*Q*[1. for i in 1:6])
display(Array(A_l)*(Q*[1. for i in 1:6]))

display("right difference")
display(Array(A_r))
display(A_r*(Q*[1. for i in 1:6]))
display(Array(A_r)*(Q*[1. for i in 1:6]))
