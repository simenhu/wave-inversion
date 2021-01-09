push!(LOAD_PATH, "./src/simutils/")
using Zygote
using Simutils
import ChainRulesCore: rrule, DoesNotExist, NO_FIELDS
using Random
using DiffEqOperators
using Plots
using LinearAlgebra
using Infiltrator

plotlyjs()
Random.seed!(1)

deriv_function(x) = sum(mutation_testing(x))

x_deriv_test = 5.

##
Array(Zygote.gradient(deriv_function, x_deriv_test)[1])

## Testing derivative of Derivative operator

## Defining constants for string property
T = 100.0 # N
μ = 0.01 # Kg/m
sim_time = (0.0, 0.15)
string_length = 2*pi
dx = 0.01
number_of_cells = Int(div(string_length, dx))

# Making inversion data

# Defining constants for time property
Δt = 0.001
t_vector = sim_time[1]:Δt:sim_time[2]
frequency = 50

f_excitation = gaussian_excitation_function(100, 0.005, sim_time, 0.03, 0.017)
internal_positions = internal_node_positions(0, string_length, number_of_cells)

## Initial conditions
# initial_position = sin.((2*pi/string_length)*internal_positions)
u_0 = make_initial_condition(number_of_cells)
# a_coeffs = b_coeffs = make_material_coefficients(number_of_cells, [sqrt(T/μ), 1.5*sqrt(T/μ), 0.5*sqrt(T/μ)], [[1], [300], [450]])
a_coeffs = b_coeffs = make_material_coefficients(number_of_cells, [sqrt(T/μ)], [[1]])
Θ = (a_coeffs, b_coeffs)

## Define ODE function
f = general_one_dimensional_wave_equation_with_parameters(string_length, number_of_cells, Θ, excitation_func=[f_excitation], excitation_positions=[100], pml_width=60)

t=0.0
# grad = Zygote.gradient(coeffs -> sum(f(u_0, (coeffs, b_coeffs), t)), a_coeffs)

## Test with simpler function

number_of_cells_2 = 100
# u_2 = rand(number_of_cells_2+2)  # adding two here to make up for the missing ghost nodes
u_2 = sin.(internal_node_positions(0, 2*pi, number_of_cells_2))
coeffs_2 = [1.0 for i in 1:number_of_cells_2]

function simple_adjont_test_function()
    function du(x, coeffs)
        Ax = RightStaggeredDifference{1}(1, 4, dx, number_of_cells_2, coeffs)
        Q = Dirichlet0BC(Float64)
        # @infiltrate
        return Ax*(Q*x), Ax
    end
end
f_2 = simple_adjont_test_function()

grad = Zygote.gradient(coeffs -> sum(f_2(u_2, coeffs)[1]), coeffs_2)
plot(grad[1])

res, A = f_2(u_2, coeffs_2)
plot(res)