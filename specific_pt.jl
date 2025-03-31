
energy(x) = -5*exp(-(x+2)^2/0.5)-exp(-2*(x-2.5)^2)+x^2/30


function pt_gaussians(Nexchanges, Ntemps, Nsteps, init_state, T0, λ)
    temperatures = T0 * λ.^(range(0, Ntemps-1, length=Ntemps))
    @assert length(init_state) == Ntemps
    curr_state = init_state
    chain = zeros(Nexchanges, Ntemps, Nsteps)
    for exchange in 1:Nexchanges
        # Sampling step
        for temp in 1:Ntemps
            # samples
            for step in 1:Nsteps
                # sample
                next_state = curr_state[temp] + randn()
                # Metropolis step
                ΔE = energy(next_state) - energy(curr_state[temp])
                if ΔE < 0
                    curr_state[temp] = next_state
                elseif rand() < exp(-ΔE)
                    curr_state[temp] = next_state
                end
                chain[exchange, temp, step] = curr_state[temp]
            end
        end
        # Exchange step
        for temp in 1:Ntemps-1
            ΔE_exchange = (energy(curr_state[temp]) - energy(curr_state[temp+1])) * (1/temperatures[temp] - 1/temperatures[temp+1])
            if ΔE_exchange < 0
                curr_state[temp], curr_state[temp+1] = curr_state[temp+1], curr_state[temp]
            elseif rand() < exp(-ΔE_exchange)
                curr_state[temp], curr_state[temp+1] = curr_state[temp+1], curr_state[temp]
            end
            
        end
    end
end
