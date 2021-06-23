##
push!(LOAD_PATH, "./src/simutils/")
using DiffEqOperators
using DifferentialEquations
using DiffEqFlux
using OrdinaryDiffEq
using Optim
using Flux
using DiffEqSensitivity
using Zygote

using LinearAlgebra
using Plots
using DataInterpolations
using TimerOutputs
using BenchmarkTools
using FiniteDifferences
using Infiltrator
using DelimitedFiles
using Simutils

# plotlyjs()
pyplot()

## Defining constants for string property
T = 100.0 # N
μ = 0.01 # Kg/m
sim_time = (0.0, 0.14)
string_length = 2*pi
dx = 0.01
number_of_cells = Int(div(string_length, dx))
dt = 0.0001
abstol = 1e-8
reltol = 1e-8
frequency_weight = 1.0e-13

## Making inversion data
# Defining constants for time property
Δt = 0.001
t_vector = sim_time[1]:Δt:sim_time[2]
frequency = 50

f_excitation = gaussian_excitation_function(100, 0.005, sim_time, 0.03, 0.017)
internal_positions = internal_node_positions(0, string_length, number_of_cells)

## Initial conditions
u_0 = make_initial_condition(number_of_cells)
coeff_background = make_material_coefficients(number_of_cells, [sqrt(T/μ)], [[1]])

a_coeffs = copy(coeff_background)
b_coeffs = make_material_coefficients(number_of_cells, [sqrt(T/μ), sqrt(T/μ)*1.5], [[1], [400]])
p =  hcat(a_coeffs, b_coeffs)


## Define ODE function
f = general_one_dimensional_wave_equation_with_parameters(string_length, number_of_cells, function_array=[f_excitation], excitation_positions=[100], pml_width=60)
prob = ODEProblem(f, u_0, sim_time, p=p)

## Simulate true parameters
to = TimerOutput()
solvers =  [Tsit5(), TRBDF2(), Rosenbrock23(), AutoTsit5(Rosenbrock23()), Midpoint(), Vern7(),
            KenCarp4(), ORK256(), ParsaniKetchesonDeconinck3S32(), SSPRK22()]

solver = solvers[8]
sol = @timeit to "simulation" solve(prob, solver, save_everystep=true, p=p, abstol=abstol, reltol=reltol, dt=dt)
time_seconds = sol.t .* dt

## Define pertrbated parameters
a_coeffs_pertrbated = copy(coeff_background)
b_coeffs_perturbated =  make_material_coefficients(number_of_cells, [sqrt(T/μ), sqrt(T/μ)*1.45], [[1], [400]])
p_perturbated =  hcat(a_coeffs_pertrbated, b_coeffs_perturbated)
wrong_sol = solve(prob, solver, save_everystep=true, p=p_perturbated, abstol=abstol, reltol=reltol, dt=dt)


## Plot time domain response
p11 = plot(time_seconds, sol[receiver_position, :], label="True model")
p12 = plot(time_seconds, wrong_sol[receiver_position, :], label="Perturbated model")
p13 = plot(time_seconds, wrong_sol[receiver_position, :] - sol[receiver_position, :], label="Difference")
p1 = plot(p11, p12, p13, layout=(3, 1), link = :x, xlabel="time [s]", xformatter=:scientific)
display(p1)
# savefig("figures/experiment_with_one_transition/time_domain_reflection.eps")

## Define loss function

solution_time = sol.t
function predict(Θ)
    Array(solve(prob, solver, p=Θ, saveat=solution_time; sensealg=InterpolatingAdjoint(),  abstol=abstol, reltol=reltol, dt=dt))
end


# loss(Θ) = error_position_loss(predict, sol, Θ, 100)
loss(Θ) = error_position_loss_spatial_regularization(predict, sol, Θ, 100, frequency_weight)


## Test that the loss function makes sense
loss_with_correct_param = loss(p)[1]
loss_with_wrong_param = loss(p_perturbated)[1]
display("Loss with correct param: $(loss_with_correct_param)")
display("Loss with wrong param: $(loss_with_wrong_param)")


## Gradient of initial model
# @profview global grad_coeff = @timeit to "gradient calculation" Zygote.gradient(Θ -> loss(Θ)[1], Θ_start)
grad_coeff_zygote = @timeit to "gradient" Zygote.gradient(Θ -> loss(Θ)[1], p_perturbated)[1]
p1 = plot(grad_coeff_zygote[:,1], label="a_coeffs", size=(700, 350))
p2 = plot!(grad_coeff_zygote[:, 2], label="b_coeffs", xlabel="position")
display(p2)
# savefig("figures/experiment_with_one_transition/start_model_gradient.eps")
display(to)


## Optimize with ADAM
# Reset parameters
LOSS = []
PRED = []
PARS = []

cb = function(params, l, pred)
    display(l)
    append!(PRED, [pred])
    append!(LOSS, l)
    append!(PARS, [params])
    false
end

opt_res = DiffEqFlux.sciml_train(Θ -> loss(Θ), p_perturbated, ADAM(0.1), cb = cb, maxiters = 30, allow_f_increases = true)
opt_sol = solve(prob, solver, save_everystep=true, p=opt_res.u, abstol=abstol, reltol=reltol, dt=dt)

## Plot resulting coefficients from ADAM optimizer
p11 = plot(p, label="starting coefficients", legend=:topleft)
p12 = plot!(p_perturbated, label="true coefficients")
p13 = plot!(opt_res.u, label="Optimized coefficients")

p21 = plot(opt_res.u - p_perturbated, label="Change from initial model", legend=:topleft)

p1 = plot(p11, p21, layout=(2, 1), size=(700, 350), xlabel="position", link=:x)
display(p1)
savefig("figures/experiment_with_one_transition/optimized_coefficients.eps")

## Plot result time domain response
p11 = plot(time_seconds, sol[receiver_position, :], label="True time response")
p12 = plot!(time_seconds, wrong_sol[receiver_position, :], label="Perturbated time response")
p13 = plot!(time_seconds, opt_sol[receiver_position, :], label="Optimized time response")

p21 = plot(time_seconds, wrong_sol[receiver_position, :] - sol[receiver_position, :], label="Initial error")
p22 = plot!(time_seconds, opt_sol[receiver_position, :] - sol[receiver_position, :], label="Optimized error")
p23 = plot!(time_seconds, PRED[3][receiver_position, :] - sol[receiver_position, :], label="Intermediate error")

p1 = plot(p11, p21, layout=(2, 1), size=(700, 350), xlabel="time [s]", link=:x)
display(p1)
savefig("figures/experiment_with_one_transition/optimized_time_response.eps")

## Plot development of prediction
pred_array = hcat([pred[receiver_position, :] for pred in PRED]...)
pred_array_difference = pred_array .- sol[receiver_position, :]
h1 = heatmap(pred_array_difference, ylabel="time [s]")
p12 = plot(LOSS, ylabel="loss", ylims=(0, max(LOSS...)))
p1 = plot(h1, p12, layout=(2, 1), xlabel="Iteration", link=:x, legend=false)
display(p1)
savefig("figures/experiment_with_one_transition/time_response_development.eps")


## Plot development of coefficients
pars_array = hcat([pars[:, 2] for pars in PARS]...)
pars_array_difference = pars_array .- p_perturbated[:, 2]
h1 = heatmap(pars_array_difference, label="coefficients")
p12 = plot(LOSS, label="loss")
p1 = plot(h1, p12, layout=(2, 1), link=:x)
display(p1)


## Optimize with increasing frequencies
current_b_coeff = b_coeffs_perturbated
for freq in 1:15
    global current_b_coeff

    res = DiffEqFlux.sciml_train(b_coeffs -> loss_with_freq([a_coeffs_pertrbated b_coeffs], freq), current_b_coeff, BFGS(), cb = cb, maxiters = 10, allow_f_increases = false)
    current_b_coeff = res.u
    display(plot!(current_b_coeff, label="iteration-$(freq) coefficients"))
end



## Simulate system with new solution

optimized_sol = solve(prob, solver, save_everystep=true, p=[a_coeffs_pertrbated res.u], abstol=abstol, reltol=reltol, dt=dt)

## Plot time signal after optimization
p1 = plot(wrong_sol[receiver_position, :] - sol[receiver_position, :], label="before optimization")
plot!(optimized_sol[receiver_position, :] - sol[receiver_position, :], label="after optimization")
display(plot(p1))

## Plot coefficient after optimization
p1 = plot(b_coeffs, label="Correct coeffs")
plot!(b_coeffs_perturbated, label="Start coeffs")
plot!(res.u, label="Optimized coeffs")
display(plot(p1, legend=:topleft))

