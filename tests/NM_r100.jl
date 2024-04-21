# Computational time versus N/M for various choices of M and r = 100.
# See Figure 6(d).

include("../FredConv.jl")
include("HTconv.jl")
using BenchmarkTools, LinearAlgebra

# The value of M
M = [512 2048 4096];
# The value of r
R = 100;
# The value of N
N = [10 12 14 16 18 21 24 27 31 36 41 48 55 63 72 83 96 110 127 146 168 194 223 256 295 340 391 450 518 597 687 791 911 1049 1207 1390 1600 1843 2121 2443 2812 3238 3728 4292 4942 5690 6552 7544 8686 10000 12915 16681 21544 27826 35938 46416 59948 77426 100000];

l = length(N)
t_HT = zeros(length(N), 3)       # the CPU time of H-T method
t_NEW = zeros(length(N), 3)      # the CPU time of new method

for m = 1:3
    # Generate a random coefficient of length M[m]
    fe = rand(M[m]);
    NN = Int.((M[m]/512)*N); 
    
    for i = 1:1:l
        # Use Benchmark to record the CPU time of the new method
        t1 = @benchmark FredConv.fred_conv($fe, $R)
        tmin1 = minimum(t1).time ./ 1e9
        t_NEW[i,m] = tmin1

        # Generate a random coefficient of length NN[i]
        ge = rand(NN[i])
        # Use Benchmark to record the CPU time of the H-T method
        t2 = @benchmark HTconv.fred_HT($fe, $ge, $R)
        tmin2 = minimum(t2).time ./ 1e9
        t_HT[i,m] = tmin2
    end
end
