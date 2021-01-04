export simple_test_func, excitation_energy_plot, set_standard_plot_properties, animate_solution

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

function plot_solution_heatmap(sol)
end

"""
Make animation of 1D-wave equation
"""
function animate_solution(sol, a_coeffs, b_coeffs, sim_time, time_resolution)

    material_height_samples = 100
    internal_nodes = length(a_coeffs)
    internal_positions = 1:internal_nodes

    a_materials = zeros(material_height_samples, internal_nodes)
    for i in 1:material_height_samples
        a_materials[i, :] .= a_coeffs
    end

    b_materials = zeros(material_height_samples, internal_nodes)
    for i in 1:material_height_samples
        b_materials[i, :] .= b_coeffs
    end

    max_value = maximum(sol[:,:])
    min_value = minimum(sol[:,:])
    display(max_value)

    anim = @gif for t = sim_time[1]:time_resolution:sim_time[2]
        plot(internal_positions, sol(t).x[1], legend=false, ylims=(min_value, max_value))
        heatmap!(internal_positions,range(0, max_value, length=material_height_samples), a_materials) # plotting a_coeffs on top
        heatmap!(internal_positions,range(min_value, 0, length=material_height_samples), b_materials) # plotting b_coeffs on bottom
    end

    return anim
end
