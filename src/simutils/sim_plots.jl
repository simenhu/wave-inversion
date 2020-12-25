export simple_test_func, excitation_energy_plot, set_standard_plot_properties

using Plots

function simple_test_func()
    display("Test string to print")
end

function set_standard_plot_properties()
    plotlyjs()
end

function excitation_energy_plot(sol, energy, excitation_func, sim_time, time_res; solver_name="Nothing")
    state_dim = size(sol.u[1].x[1])[1]
    time_vector = sim_time[1]:time_res:sim_time[2]
    excitation_wave = [excitation_func(t) for t in time_vector]
    simulated_sol = zeros(state_dim, length(time_vector))

    display(size(simulated_sol))

    for i in eachindex(time_vector)
        simulated_sol[:, i] = sol(time_vector[i]).x[1]
    end
    
    energy_vector = energy.(time_vector)
    clims = [-3.0, 3.0]

    title_plot = plot(title = solver_name, grid = false, showaxis = false, bottom_margin = -200Plots.px)
    excitation_plot = plot(time_vector, excitation_wave, title = "excitation signal")
    state_plot = heatmap(time_vector, 1:state_dim, simulated_sol, title = "simulated")
    energy_plot = plot(time_vector, energy_vector, title = "energy")
    display(plot(title_plot, excitation_plot, state_plot, energy_plot, layout = (4, 1), size=(1400, 900), link = :x, plot_title=solver_name))
end