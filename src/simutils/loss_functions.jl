export error_loss, error_position_loss, state_sum_loss, energy_flux_loss_function, error_position_frequency_loss, error_position_frequency_energy_loss
export error_position_loss_spatial_regularization

using FFTW

## Loss funcitons
function error_loss(pred_func, sol, Θ)
    pred = pred_func(Θ)
    l = pred - sol
    return sum(abs2, l), pred
end

function error_position_loss(pred_func, sol,  Θ, position)
    pred = pred_func(Θ)
    l = pred[position, :] - sol[position, :]
    return sum(abs2, l), pred
end

function state_sum_loss(pred_func, sol, Θ)
    pred = pred_func(Θ)
    return sum(pred), pred
end

function energy_flux_loss_function(pred_func, sol, Θ, position)
    pred = pred_func(Θ)
    l = (pred[position, :] - sol[position, :]).^2
    return sum(abs2, l), pred
end

function error_position_frequency_loss(pred_func, sol, Θ, position, upper_frequency)
    pred = pred_func(Θ)
    s = Complex.(sol[position, :])
    pred_complex = Complex.(pred[position, :])
    l_diff = pred_complex - s
    l = FFTW.fft(l_diff)[1:upper_frequency]
    return sum(abs2, l), pred
end

function error_position_frequency_energy_loss(pred_func, sol, Θ, position, upper_frequency)
    pred = pred_func(Θ)
    s = Complex.(sol[position, :])
    pred_complex = Complex.(pred[position, :])
    l_diff = pred_complex - s
    l = FFTW.fft(l_diff)[1:upper_frequency].^2
    return sum(abs2, l), pred
end

function error_position_loss_spatial_regularization(pred_func, sol, Θ, position, frequency_weight)
    pred = pred_func(Θ)
    l_time = pred[position, :] - sol[position, :]
    Θ_complex = Complex.(Θ)
    l_coeff = abs.(FFTW.fft(Θ_complex))
    coeff_length = div(length(l_coeff), 2)
    l_coeff = l_coeff[1:coeff_length]
    scale_array = 1:length(l_coeff)
    # @infiltrate
    return sum(abs2, l_time) + frequency_weight*sum(scale_array .* l_coeff), pred
end