using Base.Threads

function parallel_test()
    nthreads = Threads.nthreads()
    println("Number of threads: $nthreads")
    results = Vector{Int}(undef, nthreads)

    @time @threads for i in 1:nthreads
        results[i] = i
        sleep(1)
    end
    println("Results: $results")
end

N_threads = 4

if nthreads() != N_threads  # Check if it's single-threaded
    println("Running with $N_threads threads...")

    # Invoke the script with 4 threads
    run(`julia --threads $N_threads $(@__FILE__)`)
    
else
    println("Already running with the correct number of threads: ", nthreads())

    parallel_test()
end