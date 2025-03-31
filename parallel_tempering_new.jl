

function parallel_tempering(exchanges, PT_Parameters::PTParams, sampling_parameters::Union{Dict, OrderedDict}, energy, initial_guess,args...)
    NTemps = PT_Parameters.NTemps
    Nexchanges = PT_Parameters.Nexchanges
    TMax = PT_Parameters.TMax
    TMin = PT_Parameters.TMin
    # temperatures = range(TMin, TMax, length=NTemps)
    # TN=T0*λ^N
    # TODO: we can have an arbitrary way of choosing the temperatures, Let the user decide.
    temperatures = TMax * (TMin/TMax).^(range(0, NTemps-1, length=NTemps))
    # n = range(0, NTemps-1, length=NTemps)/(NTemps-1)
    # temperatures = TMax.^(1 .-n).*TMin.^n
    plot(temperatures)
    title("Tmin=$(minimum(temperatures)) Tmax=$TMax")
    # logscale in y axis
    yscale("log")
    savefig("out/temperatures.png")
    clf()
    x = initial_guess*ones(NTemps)
    # x = x .+ randn(NTemps)
    d = Normal(0,1)
    x = x .+ rand(d, NTemps)
    @info "Temperatures: $temperatures"
    @info "Initial guess: $x"
    total_chains = zeros(Nexchanges+1,NTemps, NSteps)
    total_chains[1,:,1] = x
    @showprogress for step in 1:PTParameters.Nexchanges+1
        # Sampling step
        Threads.@threads for i in 1:NTemps # parallel!!!!!!!

            # TODO: can i change step characteristics to explore the space better?
            # propose a state and metropolis accept reject. split into step choosing and metropolis accepting. Thought only to have energies, 
            # so everything is an exponential
            x[i], total_chains[step ,i, :] = sampling_function(x[i], energy, sampling_parameters, args...)
        end
        # Exchange step
        for i in 1:NTemps-1 # starting from highest temperature, going to lowest
            # exchange for temperature i and i-1
            ΔE_exchange = (energy(total_chains[step, i, end], args...) - f(total_chains[step, i+1, end], args...)) * (1/temperatures[i] - 1/temperatures[i+1])
            if ΔE_exchange < 0
                x[i], x[i+1] = x[i+1], x[i]
            elseif rand() < exp(-ΔE_exchange)
                x[i], x[i+1] = x[i+1], x[i]
            end
        end
    end
    return total_chains[end,end,end], total_chains, exchanges
end

function sampling_function(x, energy, sampling_parameters, args...)
    NSteps=sampling_parameters["NSteps"]
    StepSize=sampling_parameters["StepSize"]
    GaussianStep=sampling_parameters["GaussianStep"]
    x = initial_guess
    chain = zeros(NSteps)
    d = Normal(0,StepSize)
    for i in 1:Nsteps
        if GaussianStep
            x_new = x + rand(d)
        else
            x_new = x + step_size*rand()
        end
        ΔE = energy(x_new, args...) - energy(x,args...)
        if ΔE < 0
            x = x_new
        elseif rand() < exp(-ΔE/T)
            x = x_new
        end
        chain[i] = x
    end
    return x, chain
end