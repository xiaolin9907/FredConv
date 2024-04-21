# Computational time versus M for various choices of r = 2.
# See Figure 5(b).

include("../FredConv.jl")
include("HTconv.jl")
using BenchmarkTools, LinearAlgebra

# The value of M
M = [100 108 115 124 133 142 152 163 175 188 201 216 232 248 266 285 306 328 352 377 404 433 465 498 534 573 614 658 706 757 812 870 933 1000 1073 1150 1233 1322 1418 1520 1630 1748 1874 2010 2155 2311 2478 2657 2849 3054 3275 3512 3765 4038 4329 4642 4978 5337 5723 6136 6580 7055 7565 8112 8698 9327 10000 10723 11498 12329 13220 14175 15200 16298 17476 18739 20093];

l = length(M)
t_HT = zeros(l ,4)     # the CPU time of H-T method
t_NEW = zeros(l, 4)    # the CPU time of new method
# The ratio of N to M
rat = [0.5 1 2];
# The value of r
R=2;
for j = 1:3
    for i = 1:1:l
        # Generate a random coefficient of length M[i]
        fe = rand(M[i]);

        # Use Benchmark to record the CPU time of the new method
        t1 = @benchmark FredConv.fred_conv($fe, $R)
        tmin1 = minimum(t1).time ./ 1e9
        t_NEW[i,j] = tmin1

        # Generate a random coefficient of length N
        N = ceil.(Int, rat[j]*M[i])
        ge = rand(N)
        # Use Benchmark to record the CPU time of the H-T method
        t2 = @benchmark HTconv.fred_HT($fe, $ge, $R)
        tmin2 = minimum(t2).time ./ 1e9
        t_HT[i,j] = tmin2
    end
end