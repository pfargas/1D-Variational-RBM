function metropolis_step(params, gaussian_step, step_params::Union{Dict,OrderedDict})
    new_params = parmas + randn(length(params), step_params["sigma"])
    return params
end
