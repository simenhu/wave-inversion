export gaussian_excitation_function

using DataInterpolations


"""
Function for excitating simulation with gausian pulse with centre frequency of f0
"""
function gaussian_excitation_function(f0, sigma, tspan, t0, amplitude=1.0)
    safety_margin = 10
    Δt = 1/(2*f0*safety_margin)
    t_vector = tspan[1]:Δt:tspan[2]

    gaussian_pulse(t) = amplitude * cos(2*pi*f0*(t - t0))*(exp(-(t-t0)^2/(2*sigma^2)))/sqrt(2*pi*sigma^2)
    excitation_vector = gaussian_pulse.(t_vector)
    f = LinearInterpolation(excitation_vector, t_vector)
    return f
end
