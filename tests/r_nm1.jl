# Computational time versus r for various choices of M and N/M = 1.
# see Figure 7(c).

include("../FredConv.jl")
include("HTconv.jl")
using BenchmarkTools, LinearAlgebra

# The value of M
M = [128 512 2048 4096];
# The ratio of N to M
N = M;
# The value of r
R = [1 2 3 4 5 7 9 11 14 18 23 30 38 48 62 78 100]

l = length(R)
t_HT = zeros(length(R), 4)       # the CPU time of H-T method
t_NEW = zeros(length(R), 4)      # the CPU time of new method
for m = 1:4
    # Generate a random coefficient of length M[m]
    fe = rand(M[m])
    # Generate a random coefficient of length N[m]
    ge = rand(N[m])
    for r = 1:1:l
        # Use Benchmark to record the CPU time of the new method
        t1 = @benchmark FredConv.fred_conv($fe, $R[$r])
        tmin1 = minimum(t1).time ./ 1e9
        t_NEW[r,m] = tmin1

        # Use Benchmark to record the CPU time of the H-T method
        t2 = @benchmark HTconv.fred_HT($fe, $ge, $R[$r])
        tmin2 = minimum(t2).time ./ 1e9
        t_HT[r,m] = tmin2
    end
end