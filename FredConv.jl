module FredConv

export fred_conv, FS_col, FS_row, Trans_col, Trans_row
using LinearAlgebra
function fred_conv(fe, r)
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

    # Type of fe
    Ty = eltype(fe)
    r = Ty(r)
    
    # Ensure that r > 0
    if r <= 0
        error("r should be larger than 0.")
    end

    # Length of fe
    n = length(fe) 

    # Initialize the output matrix
    T = zeros(Ty, n, n)

    # We bifurcate for two cases: r >= 1 and r < 1, constructing the matrix T in different ways:
    # 1. when r >= 1
    if r >= 1
        # Constructing the first two columns by Theorem 2.5 and Theorem 2.6
        L1, L2 = FS_col(fe, r) 
        view(T, :, 1) .= L1
        view(T, 2:n-1, 2) .= view(L2, 2:n-1)

        # Constructing the region S1 by (2.2a) in Theorem 2.3
        for j = 3:ceil(Int, (n+1)/(r+1))
            i = floor(Int, j*r):n+1-j # i >= j * r
            view(T, i, j) .= view(T, i, j - 2) .+ r.*(2*j - 3) .* (view(T, i .- 1, j - 1)./(2*i .- 3) .- view(T, i .+ 1, j - 1)./(2*i .+ 1))
        end
        
        # Constructing the region S23 by (2.2d) in Theorem 2.3
        for i = min(n-2, n+1-floor(Int, (n+1)/(r+1))):-1:2
            j = max(2, floor(Int, i/r)):n+1-i  # i < j * r
            view(T, i, j) .= (2*i - 1)./(r.*(2*j .- 1)) .* (view(T, i + 1, j .+ 1) .- view(T, i + 1, j .- 1)) .+ (2*i - 1)/(2*i + 3) .* view(T, i + 2, j)
        end

        # Constructing the first row
        j = 2:n-1
        view(T, 1, j) .= 1 ./ (r.*(2*j .- 1)) .* (view(T, 2, j .+ 1) .- view(T, 2, j .- 1)) .+ 1/5 .* view(T, 3, j)
        view(T, 1, n) .= -1 ./ (r*(2*n - 1))*(view(T, 2, n - 1))
    end

    #2. when r < 1
    if r < 1
        # Constructing the first two columns by Theorem 2.8
        R1, R2 = FS_row(fe, r)
        view(T, 1, :) .= R1
        view(T, 2, 2:n-1) .= view(R2, 2:n-1)

        # Constructing the region S3 by (2.2b) in Theorem 2.3
        for i = 3:ceil(Int, (n+1)/(1+1/r))
            j = floor(Int, i/r):n+1-i
            view(T, i, j) .= (2*i - 1) ./ (r*(2*j .- 1)) .* (view(T, i - 1, j .- 1) .- view(T, i - 1, j .+ 1)) .+ (2*i - 1)/(2*i - 5) .* view(T, i - 2, j)
        end

        # Constructing the region S12 by (2.2c) in Theorem 2.3
        for j = min(n-2, n+1-ceil(Int, (n+1)/(1+1/r))):-1:2
            i = max(3, floor(Int, j*r)):n+1-j
            view(T, i, j) .= view(T, i, j + 2) .- r*(2*j + 1) .* (view(T, i .- 1, j + 1)./(2*i .- 3) .- view(T, i .+ 1, j + 1)./(2*i .+ 1))
        end

        # Constructing the first column
        i = 2:n-1
        view(T, i, 1) .= view(T, i, 3) .- 3*r*(view(T, i .- 1, 2)./(2*i .- 3) .- view(T, i .+ 1, 2)./(2*i .+ 1))
        view(T, n, 1) .= -r*3*view(T, n - 1, 2)/(2*n - 3)
    end
    # Return the result
    return T
end
##########################################################################################################

function inte_legendre(v)
    # Return the coefficients of the indefinite integral of a Legendre series corresponding to v. See (2.1b) in Lemma 2.1

    n = length(v)
    v = [v; 0; 0]
    iv = zeros(n + 1)
    d = collect(1:n)
    view(iv, 2:n+1) .= view(v, 1:n)./(2*d .- 1) .- view(v, 3:n+2)./(2*d .+ 3)
    view(iv, 1) .= sum(view(iv, 2:2:n+1)) - sum(view(iv, 3:2:n+1))
    return iv
end
##########################################################################################################

function p1muti(fe)
    # Multiply a Legendre series by x. See (2.1c) in Lemma 2.1

    n = length(fe)
    va = zeros(n + 1)
    if n > 1
        va[1] = fe[2] / 3
        k = 2:n-1
        va[2:n-1] = fe[1:n-2].*(k .- 1)./(2*k .- 3) .+ fe[3:n].*k./(2*k .+ 1)
        if n > 2
            va[n] = fe[n-1]*(n - 1)/(2*n - 3)
            va[n + 1] = fe[n]*n/(2*n - 1)
        else
            va[2] = fe[1]
            va[3] = 2 * fe[2]/3
        end
    else
        va = [0, 1]
    end
    return va
end
##########################################################################################################

function Trans_col(n, r)
    # Representing {P(x+1 / r+1) - P(x-1 / r+1)} in terms of {P(x/r)}. 
    # Return the transfer matrix W_L. See Lemma 2.4 for more detail.

    Ty = typeof(r)
    W = zeros(Ty, n, n)
    
    # Determine the smallest meaningful floating point number.
    q = ceil(Int64,log10(n))
    Q = 10.0^(-2*q - 16)

    # Assign the initial value of the matrix.
    W[1, 1] = 0
    if n > 1
        W[1,2] = 2/(r + 1)
    end
    if n > 2
        W[2,3] = 6*r/(r + 1)^2
    end
    if n > 3
        W[1,4] = (5 + 5*r^2)/(r + 1)^3 - 3/(r + 1)
        W[3,4] = 10*r^2/(r + 1)^3
    end
    
    if n > 4
        L = zeros(n-1)
        L[1] = 2;
        L[2] = 6;
        for k = 1:n-3
            L[k+2] = round(Int64, ((2*k + 3)/(k + 2))*( (2*k + 2)/(2*k + 1) * L[k+1] - L[k]*k/(2*k - 1)))
        end

        for m = 2:n-3

            # When i+m is even, W(i,m) = 0. 
            d = mod(m + 1, 2)
            
            # Constructing the first element in this row.
            if d == 0
                W[1,m+3] = -Ty((2*m + 3))/(m + 2)*(
                    -2*(r/(r + 1)) * W[2,m+2]/3 -
                    ((2*m + 1)/((m + 1)*(r + 1)^2) - (m + 1)/(2*m + 3) -  m^2/((m + 1)*(2*m - 1)))*W[1,m+1] +
                    (r/(r + 1))^2 * (2*m + 1)/(m + 1)*(W[1,m+1]/3 + 2*W[3,m+1]/15) -
                    (r/(r + 1))*(2*m/(m + 1))*W[2,m]/3 +
                     (m*(m - 1))/((m + 1)*(2*m - 1))*W[1,m-1]
                )
            else
                W[2,m+3] = -Ty((2*m + 3))/(m + 2)*(
                    -2*(r/(r + 1))*(W[1,m+2] + 2*W[3,m+2]/5) -
                    ((2*m + 1)/((m + 1)*(r + 1)^2) - (m + 1)/(2*m + 3) -  m^2/((m + 1)*(2*m - 1)))*W[2,m+1] +
                    (r/(r + 1))^2 * (2*m + 1)/(m + 1)*(9*W[2,m+1]/15 + 6*W[4,m+1]/35) -
                    (r/(r + 1))*(2*m/(m + 1))*(W[1,m] + 2 * W[3,m]/5) +
                     (m*(m - 1))/((m + 1)*(2*m - 1))*W[2,m-1]
                )
            end
            
            # Constructing the transfer matrix W_L by the 9-term recurrence relation (2.3) in Lemma 2.4.
            for i = 3+d:2:m
                y1 = W[i-1,m+2]*(i - 1)/(2*i - 3) + W[i+1,m+2]*i/(2*i + 1)
                y2 = Ty(((i - 1)*(i - 2)))/((2*i - 5)*(2*i - 3))*W[i-2,m+1] +
                    (((i - 1)^2)/((2*i - 1)*(2*i - 3)) + i^2 / ((2*i - 1)*(2*i + 1)))*W[i,m+1] +
                    (i*(i + 1))/((2*i + 3)*(2*i + 1))*W[i+2,m+1]
                y3 = W[i-1,m]*(i - 1)/(2*i - 3) + W[i+1,m]*i/(2*i + 1)
                
                W[i,m+3] = -Ty((2*m + 3))/(m + 2)*(
                    -2*r/(r + 1)*y1 -
                    ((2*m + 1)/((m + 1)*(r + 1)^2) -  (m + 1)/(2*m + 3) - m^2 / ((m + 1)*(2*m - 1)))*W[i,m+1] +
                    (r/(r + 1))^2 * (2*m + 1)/(m + 1)*y2 -
                    (r/(r + 1))*(2*m/(m + 1))*y3 +
                    (m*(m - 1))/((m + 1)*(2*m - 1))*W[i,m-1]
                )
                if abs(W[i,m+3]) < Q
                    W[i,m+3] = 0
                end
            end
            W[m+2,m+3] = (4*m+6)*(r/(r+1))^(m+1) /(r+1)
        end
    end
    
    return W
end
##########################################################################################################

function Trans_row(n, r)
    # Representing {P((r-x)/(r+1)) - P((-r-x)/(r+1))} in terms of {P(x)}.
    # Return the transfer matrix hat{W}_L. See Lemma 2.7 for more detail.
    
    Ty = typeof(r)
    W = zeros(Ty, n, n)
    
    # Determine the smallest meaningful floating point number.
    q = ceil(Int64,log10(n))
    Q = 10.0^(-q-16)

    # Assign the initial value of the matrix.
    W[1,1] = 0
    if n > 1
        W[1,2] = 2*r/(r + 1)
    end
    if n > 2
        W[2,3] = -6*r/(r + 1)^2
    end
    if n > 3
        W[1,4] = r*(2*r^2 - 6*r + 2)/(r + 1)^3
        W[3,4] = 10*r/(r + 1)^3
    end

    if n > 4
        for m = 2:n-3
            
            # When i+m is even, W(i,m) = 0. 
            d = mod(m + 1, 2)

            # Constructing the first two rows in W.
            if d == 0
                W[1,m+3] = Ty((2*m + 3))/((r + 1)*(m + 2))*(
                    -2*W[2,m+2]/3 +
                    (r^2 * (2*m + 1)/((m + 1)*(r + 1)) - (r + 1)*(m + 1)/(2*m + 3) - (r + 1)*m^2 / ((m + 1)*(2*m - 1)))*W[1,m+1] -
                    (2*m + 1)/((m + 1)*(r + 1))*(W[1,m+1]/3 + 2*W[3,m+1]/15) -
                    (2*m/(m + 1))*W[2, m]/3 -
                    (r + 1)*(m*(m - 1))/((m + 1)*(2*m - 1))*W[1,m-1]
                )
            else
                W[2,m+3] = Ty((2*m + 3))/((r + 1)*(m + 2))*(
                    -2*(W[1,m+2] + 2*W[3,m+2]/5) +
                    (r^2 * (2*m + 1)/((m + 1)*(r + 1)) - (r + 1)*(m + 1)/(2*m + 3) - (r + 1)*m^2 / ((m + 1)*(2*m - 1)))*W[2,m+1] -
                    (2*m + 1)/((m + 1)*(r + 1))*(9*W[2,m+1]/15 + 6*W[4,m+1]/35) -
                    (2*m/(m + 1))*(W[1,m] + 2 * W[3,m]/5) -
                    (r + 1)*(m*(m - 1))/((m + 1)*(2*m - 1))*W[2,m-1]
                )
            end
            
            # Constructing the transfer matrix hat{W}_L by Lemma 2.7.
            for i = 3+d:2:m
                y1 = W[i-1,m+2]*(i - 1)/(2*i - 3) + W[i+1,m+2]*i/(2*i + 1)
                y2 = Ty(((i - 1)*(i - 2))) / ((2*i - 5)*(2*i - 3))*W[i-2,m+1] +
                    (((i - 1)^2)/((2*i - 1)*(2*i - 3)) + i^2 / ((2*i - 1)*(2*i + 1)))*W[i,m+1] +
                    (i*(i + 1))/((2*i + 3)*(2*i + 1))*W[i+2,m+1]
                y3 = W[i-1,m]*(i - 1)/(2*i - 3) + W[i+1,m]*i/(2*i + 1)

                W[i,m+3] = Ty((2*m + 3))/((r + 1)*(m + 2))*(
                    -2*y1 +
                    (r^2 * (2*m + 1)/((m + 1)*(r + 1)) - (r + 1)*(m + 1)/(2*m + 3) - (r + 1)*m^2 / ((m + 1)*(2*m - 1)))*W[i,m+1] -
                    (2*m + 1)/((m + 1)*(r + 1))*y2 -
                    (2*m/(m + 1))*y3 -
                    (r + 1)*(m*(m - 1))/((m + 1)*(2*m - 1))*W[i,m-1]
                )
                if abs(W[i,m+3]) < Q
                    W[i,m+3] = 0
                end
            end

            W[m+2,m+3] = Ty((2*m + 3))/((r + 1)*(m + 2))*(
                -2*W[m+1,m+2]*(m + 1)/(2*(m + 2) - 3) -
                (2*m + 1)/((m + 1)*(r + 1))*(((m + 1)*(m))/((2*(m + 2) - 5)*(2*(m + 2) - 3)))*W[m,m+1]
            )
        
        end
    end
    
    return W
end
##########################################################################################################

function FS_row(fe, r)
    # Return the first two rows of T. See Theorem 2.8 for more detail.

    i_f = (r + 1).*inte_legendre(fe)  # Indefinite integral of f or U * a.
    n = length(i_f)                   # Length of the primitive function.
    
    T_l = Trans_row(n + 1, r)         # The transfer matrix hat{W}_L.
    
    v = view(T_l, 1:n, 1:n)*i_f       # (r+1) * W * U * a in Theorem 2.8.
    va = v[1:n-1]
    
    w = 0:n-2
    W = r.*(2*w .+ 1)              
    va = va./W                        # (r+1)/r * D * W * U * a which is the zeroth row
    
    f1 = p1muti(fe)                   # Multiply f by x or V * a 
    f2 = (r + 1).*inte_legendre(f1)   # (r+1) * U * V * a 
    v1 = T_l*f2                       # (r+1) * W * U * V * a 
    v1 = v1[1:end-2]
    
    v3 = p1muti(v)                    #(r+1) * V * W * U * a
    v3 = v3[1:end-2]
    
    v = ((r + 1)/r.*v1 .+ 1/r.*v3)
    v = 3*v./W                        # The first row
    
    return va, v
end
##########################################################################################################

function FS_col(fe, r)
    # Return the first two rows of T. See Theorems 2.5 and 2.6 for more detail.

    i_f = (r + 1).*inte_legendre(fe)  # Indefinite integral of f or (r+1) * U * a.
    n = length(i_f)                   # Length of the primitive function.
    
    T_l = Trans_col(n + 1, r)         # The transfer matrix W_L.
    L1 = view(T_l,1:n, 1:n)*i_f       # (r+1) * W * U * a 
    L1 = L1[1:end-1]                  # The zeroth column 
    
    f1 = p1muti(fe)                   # Multiply f by x or V * a
    f2 = (r + 1).*inte_legendre(f1)   # (r+1) * U * V * a 
    
    v1 = T_l*f2                       # (r+1) * W * U * V * a 
    v1 = v1[1:end-1]
    
    v2 = p1muti(L1)                   #(r+1) * V * W * U * aF
    L2 = -(r + 1).*v1 .+ r.*v2
    
    L2 = L2[1:end-1]                  # The first column
    
    return L1, L2
end

end