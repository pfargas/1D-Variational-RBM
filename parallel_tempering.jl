using ProgressMeter

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

struct Exchanges
    Texchange::Vector
    ExchangeStep::Vector{Int}
end

"""
# Parallel Tempering MCMC (PTMCMC)

with PTParams we define the number of temperatures, the max and min of them and generate a linspace of temperatures. Also we define how many times do we want to exchange the temperatures.
with metropolisparams we can define the number of steps that we want to do in each time we sample from metropolis (aka the steps before an exchange)
"""

function parallel_tempering(exchanges, PTParameters::PTParams, MetropolisParams::MetropolisParams, f, initial_guess,args...)
    NTemps = PTParameters.NTemps
    TMax = PTParameters.TMax
    TMin = PTParameters.TMin
    NSteps = MetropolisParams.NSteps
    StepSize = MetropolisParams.StepSize
    GaussianStep = MetropolisParams.GaussianStep

    temperatures = range(TMax, TMin, length=NTemps)
    x = initial_guess
    total_chains = zeros(Nexchanges,NTemps, NSteps)

    @showprogress for step in 1:PTParameters.Nexchanges
        # Sampling step
        chains = zeros(NTemps, NSteps)
        for i in 1:NTemps # parallel!!!!!!!
            # x, chain = metropolis(NSteps, f, x, StepSize, GaussianStep, args)
            x, chains[i, :] = metropolis_energy(NSteps, f, x, StepSize, GaussianStep, temperatures[i], args)
             #chains[1] is the chain of the first temperature, chains[2] is the chain of the second temperature, etc
        end
        # Exchange step
        for i in 1:NTemps-1 # starting from highest temperature, going to lowest
            # exchange for temperature i and i-1
            ΔE_exchange = (f(chains[i, end], args...) - f(chains[i+1, end], args...)) * (1/temperatures[i+1] - 1/temperatures[i])
            if ΔE_exchange < 0
                chains[i, :], chains[i+1, :] = chains[i+1, :], chains[i, :]
                append!(exchanges.Texchange, temperatures[i])
                append!(exchanges.ExchangeStep, i)
            else
                if rand() < exp(-ΔE_exchange)
                    chains[i, :], chains[i+1, :] = chains[i+1, :], chains[i, :]
                    append!(exchanges.Texchange, temperatures[i])
                    append!(exchanges.ExchangeStep, i)
                end
            end
        end
        total_chains[step, :, :] = chains
    end
    return total_chains[end,end,end], total_chains, exchanges

end

function flatten_chains(chains)
    new_chains = zeros(size(chains,1)*size(chains,3),size(chains,2)) # whole chain, temperature
    for i in 1:size(chains,2)
        temp_chain = zeros(1)
        for j in 1:size(chains,1)
            append!(temp_chain, chains[j,i,:])
        end
        new_chains[:,i] = temp_chain
    end
    return new_chains
end

gaussian_energy_landscape(x,args...) = -5*exp(-x^2/0.25)-exp(-(x-2.5)^2/0.75)-exp(-(x+1.5)^2/5)+x^2/10
plot_landscapes = true
using PyPlot
clf()
if plot_landscapes
    x = range(-10, 10, length=1000)
    plot(x, gaussian_energy_landscape.(x))
    savefig("out/energy_landscape.png")
    clf()
end

#lets find the min with parallel tempering

initial_guess = -15.0
TMax = 2.0
TMin = 1.0
NTemps = 4
Nexchanges = 100
NSteps = 100
StepSize = 0.01
GaussianStep = true

PTParameters = PTParams(TMax, TMin, NTemps, Nexchanges)
MetropolisParameters = MetropolisParams(NSteps, StepSize, GaussianStep)

exchanges = Exchanges([], [])

final, chains, _ = parallel_tempering(exchanges, PTParameters, MetropolisParameters, gaussian_energy_landscape, initial_guess)

chains = flatten_chains(chains)
print(chains) # LOOK INTO FLATTEN CHAINS

for i in 1:NTemps
    plot(chains[:,i,:], label="T = $(round(range(TMax, TMin, length=NTemps)[i], digits=2))")
end

title("$final")
legend()
savefig("out/chain.png")