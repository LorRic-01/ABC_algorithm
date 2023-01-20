function [opt, hive, ABC_time] = ABC(dim, f, lb, ub, g, type, cycle, ...
                n_emp, n_onl, gen, phi, maxIter, ...
                n_opt, tol, hive_i)
% ABC - Artificial Bee Colony algorithm
%   Optimization algorithm that solves mir or max problems in the form
%   arg__ f(x)  subject to:     g(x) = 0    (equality constraints)
%       x                    lb <= x <= ub  (bounds)
%
%   Problem settings:
%      dim    - problem's dimension        [-]
%      f      - (nectar) cost function     [@fun(), mex()]
%      lb     - solution lower bound       [-inf (default) | double(1, dim)]
%      ub     - solution upper bound       [ inf (default) | double(1, dim)]
%      g      - equality constraint        [none (default) | @fun(), mex()]
%      type   - optimization type          ['min' (default) | 'max']
%      cycle  - # of algorithm iterations  [100 (default)]
%
%   Colony settings:
%      n_emp    - # of employed bees                        [100 (default)]
%      n_onl    - # of onlooker bees                        [100 (default)]
%      gen      - bees initialization function              [1000*rand - 500 (default) | @fun(), mex()]
%      phi      - random function in [-1, 1]                [2*rand - 1 (default) | @fun(), mex()]
%      maxIter  - max non-update sol iter before rejection  [10 (default)]
%      hive_i   - hive initialization                       [double(n_i, dim + dim_g)]
%     
%   Solution settings
%      n_opt  - number of returned optimal sol          [10 (default)]
%      tol    - tolerance to discard returned near sol  [1 (default)]
%
%   Output:
%      opt    - optimal solutions   [double(n_opt, dim)]
%      hive   - general hive        [double(n_emp + n_onl + n_opt, dim + dim_g)]

tic
%% Internal arameters
showFig = [false, false, true];  % show figure ([init, loop, end])
nFig = 1;                       % figure number for plotting

%% Default input parameters
if nargin < 2 || isempty(dim) || isempty(f)
    error('Missing problem dimension and/or cost function')
end
if (nargin <  3) || isempty(lb),      lb = -inf(1, dim); end
if (nargin <  4) || isempty(ub),      ub = inf(1, dim); end
if (nargin <  5) || isempty(g),       g = []; end
if (nargin <  6) || isempty(type),    type = 'min'; end
if (nargin <  7) || isempty(cycle),   cycle = 100; end
if (nargin <  8) || isempty(n_emp),   n_emp = 100; end
if (nargin <  9) || isempty(n_onl),   n_onl = 100; end
if (nargin < 10) || isempty(gen),     gen = @(n, dim) 1000*rand(n, dim) - 500; end
if (nargin < 11) || isempty(phi),     phi = @() 2*rand - 1; end
if (nargin < 12) || isempty(maxIter), maxIter = 20; end
if (nargin < 13) || isempty(n_opt),   n_opt = 10; end
if (nargin < 14) || isempty(tol),     tol = 1; end

% Additional parameters
if ~isempty(g)
    dim_g = length(g(zeros(1, dim)));
    L = @(x) f(x(:, 1:dim)) + sum(x(:, dim+1:end).*g(x(:, 1:dim)), 2);
else
    dim_g = 0; L = @(x) f(x);
end

switch type
    case 'max', type = -1;
    otherwise,  type = 1;
end

%% Hive initialization
if (nargin < 15) || isempty(hive_i) || size(hive_i, 2) ~= dim + dim_g
    hive = gen(n_emp + n_onl + n_opt, dim + dim_g);
else
    hive = [hive_i(1:min(size(hive_i, 1), n_emp + n_onl + n_opt), :);
    gen(min(n_emp + n_onl + n_opt - size(hive_i, 1), 0), dim + dim_g)];
end

n_nup = zeros(n_emp + n_onl, 1);    % # of non updated iteration

% Optimal solution
hive(end-n_opt+1:end, :) = optSol(dim, hive, type, f, g, n_opt, tol);

if showFig(1), drawHive(nFig, hive, L, n_emp, n_onl, n_opt, [min(lb), max(ub)]); end

%% Algorithm
for iter = 1:cycle
    % Resample onlooker bees
    if type == -1, f_Lm = abs(L(hive(1:n_emp, :)) - min(L(hive(1:n_emp, :))) + 1);
    else, f_Lm = abs(L(hive(1:n_emp, :)) - max(L(hive(1:n_emp, :))) - 1); end

    % Choose probability and its density function
    prob = f_Lm / sum(f_Lm);
    dens_prob = cumsum(prob);
    
    indexes = zeros(n_onl, 1);
    for k = 1:n_onl
        index = find(dens_prob > rand, 1);
        indexes(k) = index;
        hive(n_emp+k, :) = hive(index, :);
    end

    % Move bees
    for i = 1:n_emp + n_onl
        % Choose randomly bee to comapre and dimansion
        k = i;
        while k == i
            k = randi(n_emp + n_onl, 1);
            j = randi(dim + dim_g, 1);
        end

        if j <= dim
            % Solution optimization
            new_sol = hive(i, :);
            new_sol(1, j) = new_sol(1, j) + phi()*(new_sol(1, j) - hive(k, j));
            [hive(i, 1:dim), check] = feasPos(new_sol(1, 1:dim), hive(i, 1:dim), ...
                                type*L(new_sol), type*L(hive(i, :)), lb, ub);
        else
            % Lagrangian variable optimization
            % Move toward the constraints
            j1 = randi(dim, 1);
            new_sol = hive(i, 1:dim);
            new_sol(1, j1) = new_sol(1, j1) + phi()*(new_sol(1, j1) - hive(k, j1));
            [hive(i, 1:dim), check] = feasPos(new_sol(1, 1:dim), hive(i, 1:dim), ...
                                sum(abs(g(new_sol))), sum(abs(g(hive(i, 1:dim)))), lb, ub);
            
            % Update Lagrangian multipliers
            alpha = min([0.01, abs(g(hive(i, 1:dim)))]);
            df = f(hive(i, 1:dim) + diag(alpha*hive(i, 1:dim))) - f(hive(i, 1:dim) - diag(alpha*hive(i, 1:dim)));
            dg = g(hive(i, 1:dim) + diag(alpha*hive(i, 1:dim))) - g(hive(i, 1:dim) - diag(alpha*hive(i, 1:dim)));

            hive(i, j) = -dg(:, j-dim)'*df/(norm(dg(:, j-dim)).^2);
        end

        if check
            n_nup(i) = n_nup(i) + 1;
        end
    end
    
    % Optimal solution
    hive(end-n_opt+1:end, :) = optSol(dim, hive, type, f, g, n_opt, tol);
    
    if showFig(2), drawHive(nFig, hive, L, n_emp, n_onl, n_opt, [min(lb), max(ub)]); end
    
    hive(n_nup >= maxIter, :) = gen(sum(n_nup >= maxIter), dim + dim_g);
    n_nup(n_nup >= maxIter, :) = zeros(sum(n_nup >= maxIter), 1);
end

opt = hive(end-n_opt+1:end, 1:dim);
ABC_time = toc;
if showFig(3), drawHive(nFig, hive, L, n_emp, n_onl, n_opt, [min(lb), max(ub)]); end

end

%% Functions
function opt = optSol(dim, hive, type, f, g, n_opt, tol)
    % Return the first n_opt optimal points of the current hive s.t. are
    % the nearest w.r.t. the constraint and best cost
    % Input:
    %   dim    - problem's dimension        [-]
    %   hive   - general hive               [double(n_emp + n_onl + n_opt, dim + dim_g)]
    %   type   - optimization type          [-1 -> maximization, 1 -> minimization]
    %   f      - (nectar) cost function     [@fun(), mex()]
    %   g      - equality constraint        [@fun(), mex()]
    %   n_opt  - # of returned optimal sol  [-]
    %   tol    - tolerance to discard       [-]
    % Output:
    %   opt - optimal solutions             [double(n_opt, dim + dim_g)]

    % Compute cost
    f_f = f(hive(:, 1:dim)); f_g = sum(abs(g(hive(:, 1:dim))), 2);

    switch type
        case -1     % maximization
            [~, index] = sort(f_f, 'descend');
        otherwise   % minimization
            [~, index] = sort(f_f, 'ascend');
    end
    hive = hive(index, :); f_g = f_g(index, :);
    [~, index] = sort(f_g, 'ascend'); hive = hive(index, :);
    
    % Discard too near solution
    opt = hive((end-n_opt + 1):end, :); opt(1, :) = hive(1, :);
    index = 2;
    for k = 1:size(hive, 1) - n_opt
        if all(sqrt(sum((opt(1:index-1, 1:dim) - hive(k, 1:dim)).^2, 2)) > tol)
            opt(index, :) = hive(k, :);
            index = index + 1;
        end

        if index > n_opt
            break;
        end
    end
end

function drawHive(nFig, hive, L, n_emp, n_onl, n_opt, minmax)
    % Plot the 2D or 3D graph of the system
    % Input:
    %   nFig    - figure #                      [-]
    %   hive    - possible solutions            [double(n, dim + dim_g)]
    %   L       - Lagrangian function           [@L()]
    %   n_emp   - # of employed bees            [-]
    %   n_onl   - # of onlooker bees            [-]
    %   n_opt   - # of returned optimal sol     [-]
    %   minmax  - printing bounds               [double(1, 2)]

    figure(nFig), hold off
    if size(hive, 2) < 2
        fplot(@(x) L(x), minmax), hold on
        scatter(hive(1:n_emp, 1), L(hive(1:n_emp, :)), 'r', 'filled')
        scatter(hive(n_emp+1:n_emp+n_onl, 1), L(hive(n_emp+1:n_emp+n_onl, :)), 'b', 'filled')
        scatter(hive((end - n_opt + 1):end, 1), L(hive((end - n_opt + 1):end, :)), 'g', 'filled')
        xlabel('x'), ylabel('L(x)'), drawnow
    else
        fsurf(@(x, y) L([x, y, hive(end-n_opt+1, 3:end)]), minmax,...
            'FaceAlpha', 0.5, 'EdgeColor', [0.3, 0.3, 0.3], 'MeshDensity', 15), hold on
        scatter3(hive(1:n_emp, 1), hive(1:n_emp, 2), L(hive(1:n_emp, :)), 'r', 'filled')
        scatter3(hive(n_emp+1:n_emp+n_onl, 1), hive(n_emp+1:n_emp+n_onl, 2), L(hive(n_emp+1:n_emp+n_onl, :)), 'b', 'filled')
        scatter3(hive((end - n_opt + 1):end, 1), hive((end - n_opt + 1):end, 2), L(hive((end - n_opt + 1):end, :)), 'g', 'filled')
        xlabel('x1'), ylabel('x2'), zlabel('L'), drawnow
    end
end

function [p, chech] = feasPos(p1, p2, f_p1, f_p2, lb, ub)
    % Return the position s.t.
    %   - if p1, p2 unfeasible  -> p = the unfeasible p
    %       (euclidian distance from the bounds)
    %   - if one is unfeasible  -> p = feasible one
    %   - if both feasible      -> p = p with min f_p
    %
    % Input:
    %   p1   - position 1           [double(1, dim)]
    %   p2   - position 2           [double(1, dim)]
    %   f_p1 - cost p1              [-]
    %   f_p2 - cost p2              [-]
    %   lb   - lower bound          [double(1, dim)]
    %   ub   - upper bound          [double(1, dim)]
    %
    % Output:
    %   p    - best position        [double(1, dim)]
    %   ch   - true if p == p2      [boolean]
    
    p = [p1; p2]; chech = false;

    % Compute the fesibility of the solution
    feas_lb = p >= lb; feas_ub = p <= ub;
    feas = all(feas_lb & feas_ub, 2);
    
    % Compute distance from the bounds
    pos_ub = p - ub; pos_lb = p - lb;
    pos_ub(feas_ub) = 0; pos_lb(feas_lb) = 0;
    d = sum(pos_ub.^2, 2) + sum(pos_lb.^2, 2);

    switch feas(1)*2 + feas(2)
        case 0
            % Both unfeasible
            if d(1) > d(2), p = p2; chech = true;
            else, p = p1; end
        case 3
            if f_p1 < f_p2, p = p1;
            else, p = p2; chech = true; end
        otherwise
            p = p(feas, :);
            chech = feas(2);
    end
end
