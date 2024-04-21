# The approximation of T[exp(x)], x \in [-2,2] obtained by Algorithm 2.1.
# See Figure 4(b) in the manuscript. 

include("../FredConv.jl")
using ApproxFun
using CSV
using DataFrames

# Constructing the Legendre approximant to exp(x) with ApproxFun
f = Fun(x -> exp(x), Legendre(-2..2))
fe = coefficients(f)

# Constructing the matrix
T = FredConv.fred_conv(fe, 1)

# Storing data
T_str = string.(T)
df_T = DataFrame(T_str, :auto)
CSV.write("output_T.csv", df_T)
