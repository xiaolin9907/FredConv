# An example borrowed from ATAP.
# f(x) = sin(1/x)sin(1/(sin(1/x))), supp(f) = [0.2885554757, 0.3549060246].
# g(x) = exp(-x^2/4t)/sqrt(4t*pi ), where t = 1e-7, supp(g) = [-0.003, 0.003].

include("../FredConv.jl")
using ApproxFun

# Domain of f
a = 0.2885554757
b = 0.3549060246
df = [a b]

# Domain of g
dg = 0.003 .* [-1, 1]
m = (df[2] - df[1])/(dg[2] - dg[1])
t = 1e-7
# Using ApproxFun to construct the Legendre series for f:
f = Fun(x -> sin(1/x)*sin(1/sin(1/x)), Legendre(df[1]..df[2]), 2000)
fe = coefficients(f)

# Using ApproxFun to construct the Legendre series for g:
g = Fun(x -> exp(-x^2/(4*t))/sqrt(4*pi*t), Legendre(dg[1]..dg[2]))
ge = coefficients(g)

# Constructing the approximation of the opperator:
T = FredConv.fred_conv(fe, m - 1)
T_Big = FredConv.fred_conv(BigFloat.(fe), m - 1)
# Caculating the Fredholm convolution:
he = ((dg[2] - dg[1])/2) .* (view(T, :, 1:length(ge))*ge)
he_Big = ((dg[2] - dg[1])/2) .* (view(T_Big, :, 1:length(ge))*ge)