# Love's integral equation: y(t) = f(t) - d/pi * int_{0}^{1} y(s)/(d^2 + (t-s)^2) dt.
# In this example, we set d = -1, f(t) = 1 + d/pi * (arctan(1-t) + arctan(t)), and the exactly solution is y(t) = 1.

include("../FredConv.jl")
using ApproxFun
using LinearAlgebra

d = -1

# Domain of kernel
dk = [-1 5]
# Using ApproxFun to construct the Legendre series for the kernel.
ker = Fun(x -> 1/(d^2 + x^2), Legendre(dk[1]..dk[2]))
# The coefficients of the Legendre series.
ker_e = coefficients(ker)

# Domain of f.
df = [0 5]
# Using ApproxFun to construct the Legendre series of f.
f = Fun(x -> 1 - (d/pi)*(atan(1 - x) + atan(x)), Legendre(df[1]..df[2]), 1000)
# The coefficients of the Legendre series.
f_e = coefficients(f)

# Domain of g.
dg = [0 1]
r = (dk[2] - dk[1])/(dg[2] - dg[1])

mn = max(length(ker_e), length(f_e))
ker_ee = zeros(mn)
ker_ee[1:length(ker_e)] .= ker_e 
f_ee = zeros(mn)
f_ee[1:length(f_e)] .= f_e 

# Constructing the approximation of the operator.
T = ((dg[2] - dg[1])/2) .* FredConv.fred_conv(ker_ee, r - 1)
n, m = size(T)
TT = I(n) .- ((d/pi) .* T)
# Solve the linear system.
g_e = TT\f_ee[1:n]