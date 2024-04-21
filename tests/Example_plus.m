% This example shows that the entrywise cancellations of the H-T method.
% See in Figure 4(a) in the manuscript.

% Using chebfun to construct the Legendre series for exp(x) in [-2, 2]
f = chebfun('exp(x)', [-2,2]);
% Restrict f on [-2, 0] and [0, 2] 
f1 = restrict(f, [-2,0]); 
f2 = restrict(f, [0,2]);

f1cheb = get(f1, 'coeffs');
f1leg = cheb2leg(f1cheb);       % Legendre coefficients of f1
f2cheb = get(f2, 'coeffs');
f2leg = cheb2leg(f2cheb);       % Legendre coefficients of f2

n = length(f1leg);
T1 = rec(f1leg, 2*length(f1leg), 1);     % The left triangle
T2 = rec(f2leg, 2*length(f2leg), -1);    % The right triangle
TT = T1 + T2;                             % plus

% log the data
TT = abs(TT);
TT = log10(TT);

% loglog diagram
figure('Position', [100, 100, 360, 440]);
h = imagesc(TT);
xlim([0.5 2*n+.5]);
ylim([0.5 3*n+.5]);
colormap(flipud(gray));
set(gca, 'XAxisLocation', 'top');
set(gca, 'XTick', 0.5:10:5*n+.5);
set(gca, 'XTickLabel', 0:10:5*n);
set(gca, 'YTick', 0.5:10:6*n+.5);
set(gca, 'YTickLabel', 0:10:6*n);
set(gca, 'LooseInset', [0,0,0,0])
set(gca, 'Position', [0.06 0.02 0.92 0.9])
cb = colorbar;
ticks = -35:5:0;
tickLabels = arrayfun(@(x) sprintf('10^{%.d}', x), ticks, 'UniformOutput', false);
cb.YTick = ticks;
cb.TickLabels = tickLabels; 
hold on
% Mark the dividing line
stp = [n+0.5 0.5];
k = round(stp(1)-stp(2));
for i = 0:k-1
    line([stp(1)-i stp(1)-i], [stp(2)+i stp(2)+i+1], 'Color', 'red', 'LineWidth', 1.5);
    line([stp(1)-i-1 stp(1)-i], [stp(2)+i+1 stp(2)+i+1], 'Color', 'red', 'LineWidth', 1.5);
end
hold off
exportgraphics(gcf, 'plus.png', 'Resolution', 300)


function T = rec(alpha, N, sgn)
        MN = length(alpha) + N;
        % Pad to make length n + 1.
        alpha = [ alpha ; zeros(MN - length(alpha), 1) ];
        if sgn == 1
            alpha = -alpha;
        end
        % S represents multiplication by 1/z in spherical Bessel space:
        e = [[1 ; 1./(2*(1:(MN-1)).'+1)], [1 ; zeros(MN-1, 1)], -1./(2*(0:MN-1).'+1)];
        S = spdiags(e, -1:1, MN, MN);
        S(1,1) = -sgn;       
        T = zeros(length(alpha), N);
        % First column of B:
        vNew = S*alpha;
        v = vNew;
        T(:,1) = vNew;       
        % The scalar case is trivial:
        if ( N == 1 )
            return
        end        
        % Second column of B:
        vNew = S*v + sgn*v;
        vOld = v;
        v = vNew;
        vNew(1) = 0;
        T(:,2) = vNew;
    
        % Loop over remaining columns using recurrence:
        for n = 3:N
            vNew = (2*n-3) * (S * v) + vOld; % Recurrence
            vNew(1:n-1) = 0;                 % Zero terms 
            T(:,n) = vNew;
            vOld = v;
            v = vNew;
        end
        for i = 1:n
            for j = i+1:n
                T(i, j) = (-1)^(i+j)*(2*i-1)/(2*j-1)*T(j,i);
            end
        end

    end