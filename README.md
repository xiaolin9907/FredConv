# Fredconv

# Return a matrix T which is a spectral approximation of the Fredholm convolution operator F[f], where F[f](g) = \int_{-1}^{1} f(x - t)*g(t) dt.
# fe is the vector containing the Legendre coefficients of f, where supp(f) = [-r-1,r+1]. ge is the vector containing the Legendre coefficients of g, where supp(g) = [-1,1].
# Then F[f](g) can be represented in the basis {P(x/r)}, and the coefficients of F[f](g) can be obtained as T*ge.
#
# Generally, if supp(f) = [a, b], supp(g) = [c, d] where b - a > d - c, then r = (b - a)/(d - c) - 1 and F[f](g) can be represented by the basis {P((2x-(a+b+c+d))/(d-c))}, and the coefficients are (d - c)/2 * T *ge.
#
# An example:
#          f(x) = sin(x), x \in [-3, 3],  g(x) = cos(x), x \in [-1, 1]. 
#          The coefficients of f and g are fe and ge.          
#          Then h(x) = \int_{-1}^{1} f(x - t)g(t) dt  can be represented by the basis {P(x/2)}, and the coefficients are (1 + 1)/2 * T * ge,
#          where T = fred_conv(fe, r), r = (3 + 3)/(1 + 1) - 1 = 2.
#
# For further details, see the manuscript "Spectral approximation of convolution opperators of Fredholm type".
