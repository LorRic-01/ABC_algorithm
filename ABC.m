function [opt, hive] = ABC(dim, f, lb, ub, g, type, cycle, ...
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

%% Default input parameters cases 
if nargin < 2
    error('Missing problem dimension and/or cost function')
end
if (nargin < 14) || isempty(tol),     tol = 1; end
if (nargin < 13) || isempty(n_opt),   n_opt = 10; end
if (nargin < 12) || isempty(maxIter), maxIter = 10; end
if (nargin < 11) || isempty(phi),     phi = @() 2*rand - 1; end
if (nargin < 10) || isempty(gen),     gen = @(n, dim) 1000*rand(n, dim) - 500; end
if (nargin <  9) || isempty(n_onl),   n_onl = 100; end
if (nargin <  8) || isempty(n_emp),   n_emp = 100; end
if (nargin <  7) || isempty(cycle),   cycle = 100; end
if (nargin <  6) || isempty(type),    type = 'min'; end
if (nargin <  5) || isempty(g),       g = []; end
if (nargin <  4) || isempty(ub),      ub = inf(1, dim); end
if (nargin <  3) || isempty(lb),      lb = -inf(1, dim); end

% Additional parameters
if ~isempty(g)
    dim_g = length(g(zeros(1, dim)));
    L = @(x) f(x(:, 1:end-dim_g)) + sum(x(:, end-dim_g+1:end).*g(x(:, 1:end-dim_g)));
else
    dim_g = 0;
    L = @(x) f(x);
end

switch type
    case 'max', type = -1;
    otherwise,  type = 1;
end

%% Hive initialization
if (nargin <  14) || isempty(hive_i) || size(hive_i, 2) ~= (dim + dim_g)
    hive = gen(n_emp + n_onl, dim + dim_g);
else
    n_emp = n_emp + max(0, size(hive_i, 1) - n_emp - n_onl);
    hive = [hive_i; gen(min(0, n_emp - n_onl - size(hive_i, 1)), dim)];
end

f_f = zeros(n_emp + n_onl, 1);
f_g = zeros(n_emp + n_onl, dim_g);
f_L = zeros(n_emp + n_onl, 1);
for k = 1:n_emp + n_onl
    f_f(k, 1) = f(hive(k, 1:dim));
    f_g(k, :) = g(hive(k, 1:dim));
    f_L(k, 1) = L(hive(k, :));
end

% Optimal solutions
opt = nan(n_opt, dim);
optSol = zeros(n_opt + n_emp + n_onl, dim);
f_optSol = type*inf(n_opt + n_emp + n_onl, 1);
g_optSol = inf(n_opt + n_emp + n_onl, 1);

% Graphs
tmp = -10:10;
[x, y] = meshgrid(tmp, tmp);
z = zeros(size(x));
for i = 1:size(z, 1)
    for j = 1:size(z, 2)
        z(i, j) = L([x(i, j), y(i, j), zeros(1, (dim_g + dim) - 2)]);
    end
end

figure(2), hold off
surf(x, y, z, 'FaceAlpha', 0.5, 'EdgeAlpha', 0.05), hold on
scatter3(hive(1:n_emp, 1), hive(1:n_emp, 2), f_L(1:n_emp), 'b', 'filled')
scatter3(hive(n_emp+1:end, 1), hive(n_emp+1:end, 2), f_L(n_emp+1:end), 'r', 'filled')
xlabel('x'), ylabel('\lambda'), drawnow

%% Algorithm
for iter = 1:cycle
    % Resample onlooker bees
    if type == -1, f_Lm = abs(f_L(1:n_emp) - min(f_L(1:n_emp)) + 1);
    else, f_Lm = abs(f_L(1:n_emp) - max(f_L(1:n_emp)) - 1); end

    prob = f_Lm / sum(f_Lm);
    dens_prob = cumsum(prob);   % probability density function

    bee_chosen = zeros(n_onl, 1);
    for k = 1:n_onl
        index1 = find(dens_prob > rand, 1);
        bee_chosen(k, 1) = index1;
    end
    hive(n_emp+1:end, :) = hive(bee_chosen, :);
    
    % Update bees solutions
    for i = 1:n_emp + n_onl
        % Choose randomly bee to compare and which dimension
        k = i;
        while k == i
            k = randi(n_emp + n_onl, 1);
            j = randi(dim + dim_g, 1);
        end

        if j <= dim
            % Solution optimization
            new_sol = hive(i, :);
            new_sol(1, j) = new_sol(1, j) + phi()*(new_sol(1, j) - hive(k, j));
            hive(i, 1:dim) = feasPos(new_sol(1, 1:dim), hive(i, 1:dim), ...
                type*L(new_sol), type*L(hive(i, :)), lb, ub);
        else
            % Lagrangian variables optimization
            hive(i, j) = inf;
            while isinf(hive(i, j)) || isnan(hive(i, j))
                % Move toward the constraints
                j1 = randi(dim, 1);
                new_sol = hive(i, :);
                new_sol(1, j1) = new_sol(1, j1) + phi()*(new_sol(1, j1) - hive(k, j1));
                g_newSol = g(new_sol(1:dim)); g_prevSol = g(hive(i, 1:dim));
                hive(i, 1:dim) = feasPos(new_sol(1, 1:dim), hive(i, 1:dim), ...
                    abs(g_newSol(j-dim)), abs(g_prevSol(j-dim)), lb, ub);
        
        
                f_g(i, :) = g(hive(i, 1:dim));
                % Update Lagrangian multipliers
                alpha = min(0.01, abs(f_g(j-dim)));
                sol_p = [hive(i, 1:j1-1), hive(i, j1)*(1+alpha), hive(i, j1+1:dim)];
                sol_m = [hive(i, 1:j1-1), hive(i, j1)*(1-alpha), hive(i, j1+1:dim)];
                df = f(sol_p) - f(sol_m); dg = g(sol_p) - g(sol_m);
        
                hive(i, j) = - df / dg(j-dim);
            end
        end

        % Update hive
        f_f(i, 1) = f(hive(i, 1:dim));
        f_g(i, :) = g(hive(i, 1:dim));
        f_L(i, 1) = L(hive(i, :));
    end

%% Solutions
optSol(n_opt+1:end, :) = hive(:, 1:dim);
f_optSol(n_opt+1:end, 1) = f_f;
g_optSol(n_opt+1:end, 1) = sum(f_g.^2, 2);

switch type
    case -1     % maximization
        [f_optSol, index1] = sort(f_optSol, 'descend');
    otherwise   % minimization
        [f_optSol, index1] = sort(f_optSol, 'ascend');
end

optSol = optSol(index1, :);
g_optSol = g_optSol(index1, :);

[g_optSol, index2] = sort(g_optSol, 'ascend');
optSol = optSol(index2, :);
f_optSol = f_optSol(index2, :);

% Discard too near solutions and save optimal one
index = 2;
opt = optSol(1, :);
for k = 1:size(optSol, 1)
    if all(sqrt(sum((opt(1:index-1, :) - optSol(k, :)).^2, 2)) > tol)
        opt(index, :) = optSol(k, :);
        
        optSol(index, :) = optSol(k, :);
        f_optSol(index, :) = f_optSol(k, :);
        g_optSol(index, :) = g_optSol(k, :);

        index = index + 1;
    end

    if index > n_opt
        break;
    end
end

end

figure(2), hold off
surf(x, y, z, 'FaceAlpha', 0.5, 'EdgeAlpha', 0.05), hold on
scatter3(hive(1:n_emp, 1), hive(1:n_emp, 2), f_L(1:n_emp), 'b', 'filled')
scatter3(hive(n_emp+1:end, 1), hive(n_emp+1:end, 2), f_L(n_emp+1:end), 'r', 'filled')
scatter3(opt(:, 1), opt(:, 2), zeros(size(opt, 1), 1), 'g', 'filled')
xlabel('x'), ylabel('\lambda'), drawnow

end

function [p, feas] = feasPos(p1, p2, f_p1, f_p2, lb, ub)
    % Return the position s.t.
    %   - if p1, p2 unfeasible  -> p = the unfeasible p
    %       (euclidian distance from the bounds)
    %   - if one is unfeasible  -> p = feasible one
    %   - if both feasible      -> p = p with min f_p
    
    p = [p1; p2];

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
            if d(1) > d(2), p = p2;
            else, p = p1; end
        case 3
            if f_p1 < f_p2, p = p1;
            else, p = p2; end
        otherwise
            p = p(feas, :);
    end
end