##
push!(LOAD_PATH, "./src/simutils/")
using Simutils
using CSV
using DataFrames
using Plots
using DataFrames
plotlyjs()
# pyplot()


## Load data from csv files 

finite_diff_grad = Array(CSV.read("/home/simen/Documents/school/masteroppgave/wave_simulation/data/finite_difference_simulation_gradient.csv", DataFrame))
zygote_grad = Array(CSV.read("/home/simen/Documents/school/masteroppgave/wave_simulation/data/zygote_simulation_gradient.csv", DataFrame))





# Make plots of gradients

p11 = plot(finite_diff_grad[:,1], label="a-coeffs, finite difference")
p12 = plot!(finite_diff_grad[:, 2], label="b-coeffs, finite difference")
p21 = plot(zygote_grad[:, 1], label="a-coeffs, Zygote")
p22 = plot!(zygote_grad[:, 2], label="b-coeffs, Zygote")
p1 = plot(p11, p21, layout=(2, 1))
display(p1)