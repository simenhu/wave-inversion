##
push!(LOAD_PATH, "./src/simutils/")
using Simutils
using CSV
using DataFrames
using Plots
using DataFrames
using PrettyTables
# plotlyjs()
pyplot()


## Load data from csv files 

finite_diff_grad = Array(CSV.read("/home/simen/Documents/school/masteroppgave/wave_simulation/data/finite_difference_simulation_gradient.csv", DataFrame))
zygote_grad = Array(CSV.read("/home/simen/Documents/school/masteroppgave/wave_simulation/data/zygote_simulation_gradient.csv", DataFrame))




## Make plots of gradients

p11 = plot(finite_diff_grad[:,1], label="a-coeffs, finite difference")
p12 = plot!(finite_diff_grad[:, 2], label="b-coeffs, finite difference")
p21 = plot(zygote_grad[:, 1], label="a-coeffs, Zygote")
p22 = plot!(zygote_grad[:, 2], label="b-coeffs, Zygote")
p1 = plot(p12, p22, layout=(2, 1))
display(p1)

## Make table of mean absolute difference
mean_absolute_difference(x1, x2) = sum(abs.(x1 - x2))/length(x1)

a_coeff_mad = mean_absolute_difference(finite_diff_grad[:,1], zygote_grad[:, 1])
b_coeff_mad = mean_absolute_difference(finite_diff_grad[:,2], zygote_grad[:, 2])

numeric_data = [a_coeff_mad, b_coeff_mad]
rows = ["a-coeffs", "b-coeffs"]

data = [rows numeric_data]
pretty_table(data, header=["variable", "finite difference"])


## Variables to plot coefficient settings

## Defining constants for string property
T = 100.0 # N
μ = 0.01 # Kg/m
sim_time = (0.0, 0.07)
string_length = 2*pi
dx = 0.01
number_of_cells = Int(div(string_length, dx))
dt = 0.0001
abstol = 1e-8
reltol = 1e-8

# Simulator settings
Δt = 0.001
t_vector = sim_time[1]:Δt:sim_time[2]
frequency = 50

# Initial coefficients
a_coeffs = make_material_coefficients(number_of_cells, [sqrt(T/μ)], [[1]])
b_coeffs = copy(a_coeffs)
Θ =  hcat(a_coeffs, b_coeffs)

## Perturbated coefficients
a_coeffs_start = make_material_coefficients(number_of_cells, [sqrt(T/μ)*1.01], [[1]])
b_coeffs_start = copy(a_coeffs_start)
Θ_start =  hcat(a_coeffs_start, b_coeffs_start)

## Make plot for coefficient gradients
p21 = plot(a_coeffs, label="a-coeffs", ylims=(99, 102))
p31 = plot!(a_coeffs_start, label="a-coeffs perturbated", legend=true)


p22 = plot(b_coeffs, label="b-coeffs", ylims=(99, 102))
p32 = plot!(b_coeffs_start, label="b-coeffs perturbated", legend=true)

p2 = plot(p21, p22, layout=(2, 1), size=(700, 350))
display(p2)