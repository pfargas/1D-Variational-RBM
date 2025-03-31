using ProgressMeter
using Logging
using Base.Threads
using Random, Distributions

io = open("log.log", "w+")
logger = SimpleLogger(io)
global_logger(logger)

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

Should I use dictionaries as inputs instead of structs? maybe dicts are more flexible...

"""

function parallel_tempering(exchanges, PTParameters::PTParams, MetropolisParams::MetropolisParams, f, initial_guess,args...)
    NTemps = PTParameters.NTemps
    Nexchanges = PTParameters.Nexchanges
    TMax = PTParameters.TMax
    TMin = PTParameters.TMin
    NSteps = MetropolisParams.NSteps
    StepSize = MetropolisParams.StepSize
    GaussianStep = MetropolisParams.GaussianStep
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
    count = 0
    @showprogress for step in 1:PTParameters.Nexchanges+1
        # Sampling step
        Threads.@threads for i in 1:NTemps # parallel!!!!!!!

            # TODO: can i change step characteristics to explore the space better?
            # propose a state and metropolis accept reject. split into step choosing and metropolis accepting. Thought only to have energies, 
            # so everything is an exponential
            x[i], total_chains[step ,i, :] = metropolis_energy(NSteps, f, x[i], StepSize, GaussianStep, temperatures[i], args...)
        end
        # Exchange step
        for i in 1:NTemps-1 # starting from highest temperature, going to lowest
            # exchange for temperature i and i-1
            ΔE_exchange = (f(total_chains[step, i, end], args...) - f(total_chains[step, i+1, end], args...)) * (1/temperatures[i] - 1/temperatures[i+1])
            ΔE_exchange = -f(total_chains[step, i, end], args...)/temperatures[i+1] - f(total_chains[step, i+1, end], args...)/temperatures[i]+f(total_chains[step, i, end], args...)/temperatures[i] + f(total_chains[step, i+1, end], args...)/temperatures[i+1]
            if ΔE_exchange < 0
                x[i], x[i+1] = x[i+1], x[i]
                count += 1
            elseif rand() < exp(-ΔE_exchange)
                x[i], x[i+1] = x[i+1], x[i]
            end
        end
    end
    print(count)
    return total_chains[end,end,end], total_chains
end

function flatten_chains(chains, temp_idx)
    temp_chain = []
    for j in 1:size(chains,1)
        append!(temp_chain, chains[j,temp_idx,:])
    end
    return temp_chain[2:end]
end

gaussian_energy_landscape(x,args...) = -5*exp(-(x+2)^2) -exp(-2*(x-5.0)^2)
plot_landscapes = true
using PyPlot
clf()
if plot_landscapes
    x = range(-10, 10, length=1000)
    plot(x, gaussian_energy_landscape.(x))
    # plot a vertical line in the minimum
    axhline(y=minimum(gaussian_energy_landscape.(x)), color="red", linestyle="--")
    savefig("out/energy_landscape.png")
    clf()
end

#lets find the min with parallel tempering

initial_guess = -1.0
λ = 0.7
TMax = 100.0
TMin = TMax*λ
NTemps = 50
Nexchanges = 100
NSteps = 100
StepSize = 0.01
GaussianStep = true

PTParameters = PTParams(TMax, TMin, NTemps, Nexchanges)
MetropolisParameters = MetropolisParams(NSteps, StepSize, GaussianStep)

exchanges = Exchanges([], [])

quadratic_energy_landscape(x,args...) = x^2

final, chains = parallel_tempering(exchanges, PTParameters, MetropolisParameters, gaussian_energy_landscape, initial_guess)

chains1 = flatten_chains(chains, 1)


# for i in 1:NTemps
#     plot(chains[:,i,:], label="T = $(round(range(TMax, TMin, length=NTemps)[i], digits=2))")
# end

# plot(chains1, label="T = $(round(range(TMax, TMin, length=NTemps)[1], digits=2))")

# chains2 = flatten_chains(chains, 2)
# plot(chains2, label="T = $(round(range(TMax, TMin, length=NTemps)[2], digits=2))")

# chains3 = flatten_chains(chains, 3)
# plot(chains3, label="T = $(round(range(TMax, TMin, length=NTemps)[3], digits=2))")

# lowest_temp_chain = flatten_chains(chains, NTemps)
# sec_lowest_temp_chain = flatten_chains(chains, NTemps-1)
# plot(sec_lowest_temp_chain[size(sec_lowest_temp_chain,1)-2*NSteps:end], label="T = $(round(range(TMax, TMin, length=NTemps)[NTemps-1], digits=2))")

# plot(lowest_temp_chain[size(lowest_temp_chain,1)-2*NSteps:end], label="T = $(round(range(TMax, TMin, length=NTemps)[NTemps], digits=2))")

# for i in NTemps-2:NTemps
temperatures = TMax * (TMin/TMax).^(range(0, NTemps-1, length=NTemps))

temp_labels = round.(temperatures, digits=2)

for i in NTemps-1:NTemps
    plot(flatten_chains(chains, i), label="T = $(temp_labels[i])")
end

title("$final")
legend()
savefig("out/chain.png")
close(io)

# using PyCall
# @pyimport matplotlib.animation as anim
# using PyPlot


# fig, ax = PyPlot.subplots(nrows=1, ncols=1, figsize=(7, 2.5))
# # ax1, ax2 = axes

# low_T_chain = flatten_chains(chains, NTemps)

# println(size(chains))

# println(size(low_T_chain))

# println()


# function make_frame(i)
#     ax.clear()
#     ax.set_title("$(i+1)")
#     ax.plot(x, gaussian_energy_landscape.(x))
#     ax.plot(low_T_chain[i+1], gaussian_energy_landscape(low_T_chain[i+1]), "ro")
    
#     # ax1.clear()
#     # ax2.clear()
#     # ax1.imshow(A[:,:,i+1, 1])
#     # ax2.imshow(A[:,:,i+1, 2])
# end

# N_iter_per_temp = size(low_T_chain,1)

# frames = [N_iter_per_temp-200:N_iter_per_temp-1]

# myanim = anim.FuncAnimation(fig, make_frame, frames=frames, interval=20, blit=false)
# myanim[:save]("test2.gif", bitrate=-1)