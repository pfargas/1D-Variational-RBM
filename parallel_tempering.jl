include("metropolis.jl")

struct PTParams
    TMax::Float64
    TMin::Float64
    NTemps::Int
    Nexchanges::Int
end

struct MetropolisParams
    NSteps::Int
    StepSize::Float32
    GaussianStep::Bool
end

"""
# Parallel Tempering MCMC (PTMCMC)

with PTParams we define the number of temperatures, the max and min of them and generate a linspace of temperatures. Also we define how many times do we want to exchange the temperatures.
with metropolisparams we can define the number of steps that we want to do in each time we sample from metropolis (aka the steps before an exchange)

"""

function parallel_tempering(PTParameters::PTParams, MetropolisParams::MetropolisParams, f, initial_guess, args...)
    NTemps = Temperatures.NTemps
    TMax = Temperatures.TMax
    TMin = Temperatures.TMin
    NSteps = MetropolisParams.NSteps
    StepSize = MetropolisParams.StepSize
    GaussianStep = MetropolisParams.GaussianStep

    temperatures = range(TMax, TMin, length=NTemps)
    x = initial_guess
    chains = zeros(NTemps, NSteps)
    for step in 1:PTParameters.Nexchanges
        for i in 1:NTemps # parallel!!!!!!!
            x, chain = metropolis(NSteps, f, x, StepSize, GaussianStep, args)
            chains[i, :] = chain #chains[1] is the chain of the first temperature, chains[2] is the chain of the second temperature, etc
        end
        # exchange
    end
    

end
