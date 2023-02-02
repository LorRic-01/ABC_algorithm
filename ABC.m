function [opt, hive, ABC_time] = ABC(dim, f, lb, ub, g, ...
                n_emp, n_onl, cycle, gen, phi, maxIter,...
                n_opt, tol, opts, hive_i)
% ABC - Artificial Bee Colony algorithm with Augmented Lagrangian Method for
% Equality-Constrained Optimization
%   Optimization algorithm that solves min problems in the form
%   argmin f(x)  subject to:     g(x) = 0    (equality constraints)
%       x                     lb <= x <= ub  (bounds)
%
%   Problem settings:
%      dim      - problem's dimension         [-]
%      f        - (nectar) cost function      [@fun(), mex()]
%      g        - equality constraints        [empty (default) | @fun(), mex()]
%      lb       - solution lower bound        [-inf (default) | double(1, dim)]
%      ub       - solution upper bound        [ inf (default) | double(1, dim)]
%      cycle    - # of algorithm iterations   [100 (default)]
%
%   Colony settings:
%      n_emp    - # of employed bees                            [100 (default)]
%      n_onl    - # of onlooker bees                            [100 (default)]
%      gen      - bees initialization function                  [200*rand - 100 (default) | @fun(), mex()]
%      phi      - random function in [-1, 1]                    [2*rand - 1 (default) | @fun(), mex()]
%      maxIter  - max non-update sol iter before rejection      [25 (default)]
%      hive_i   - hive initialization                           [double(n_i, m)]
%
%   Results settings:
%      n_opt    - number of optimal solutions   [10 (default)]
%      tol      - min dist between opt sols     [1 (default)]
%
%   Options
%      opts     - printing options                                [struct]
%                  'nFig'     - figure # for plotting             [1 (default)]
%                  'showFig'  - show figure ([init, loop, end])   [[false, false, false] (default)]
%                  'v'        - verbose                           [true (default) | on]
%
%   Output:
%      opt      - optimal solutions   [double(n_opt, dim)]
%      hive     - general hive        [double(n_emp + n_onl + n_opt, dim + dim_g)]
%      ABC_time - sol. comp. time     [double]

%% Default parameters
if isempty(dim) || isempty(f)
    error('Missing problem dimension and/or cost function')
end

if isempty(lb), lb = -inf(1, dim); end
if isempty(ub), ub = inf(1, dim); end
if isempty(n_emp), n_emp = 100; end
if isempty(n_onl), n_onl = 100; end
if isempty(cycle), cycle = 100; end
if isempty(gen), gen = @(n) 200*rand(n, dim) - 100; end
if isempty(phi), phi = @(n) 2*rand(1, n) - 1; end
if isempty(maxIter), maxIter = 25; end
if isempty(n_opt), n_opt = 10; end
if isempty(tol), tol = 1; end
if isempty(opts)
    opts = struct('nFig', 1, 'showFig', [false, false, false], 'v', true);
end

% Additional parameters
if isempty(g), dim_g = 0;
else
    dim_g = length(g(zeros(1, dim)));
    gen = @(n) [gen(n), 10*rand(n, dim_g) - 5];
    L = @(x) f(x(1:dim)) + sum(x(dim+1:end).*g(x(1:dim)));
end

%% Hive initialization
if ~isempty(hive_i) && size(hive_i, 2) == dim + dim_g
    n_bees = n_emp + n_onl + n_opt;
    hive = [hive_i(1:min(n_bees, size(hive_i, 1)), :); ...
        gen(max(n_bees - size(hive_i, 1), 0))];
else
    hive = gen(n_emp + n_onl + n_opt);
end
n_nup = zeros(size(hive, 1), 1);

% Cost function eval
f_f = zeros(size(hive, 1), 1); f_g = zeros(size(hive, 1), max(dim_g, 1));
f_dL = zeros(size(hive, 1), 1);
time = [0, 0, 0];
for i = 1:size(hive, 1)
    tic, f_f(i, :) = f(hive(i, 1:dim)); time(1) = time(1) + toc;
    if dim_g
        tic, f_g(i, :) = g(hive(i, 1:dim)); time(2) = time(2) + toc;
        tic, f_dL(i, :) = sum(discDiff(L, hive(i, :), dim).^2) + sum(f_g(i, :).^2); time(3) = time(3) + toc;
    end
end

% Optimal solution
if ~dim_g
    [hive(end-n_opt+1:end, :), best_index] = optSol(f_f, hive, dim, n_opt, tol);
    f_f(end-n_opt+1:end, :) = f_f(best_index, :);
else
    [hive(end-n_opt+1:end, :), best_index] = optSol(f_dL, hive, dim, n_opt, tol);
    f_g(end-n_opt+1:end, :) = f_g(best_index, :);
    f_dL(end-n_opt+1:end, :) = f_dL(best_index, :);
end
    
%% Plot and verbose
if opts.showFig(1)
    if ~dim_g, drawHiveFun(opts.nFig, dim, hive, f, f_f, n_emp, n_onl, ...
                    n_opt, lb, ub);
    else, drawHiveLag(opts.nFig, dim, hive, g, L, f_dL, n_emp, n_onl, n_opt, lb, ub)
    end
end

if opts.v
    if dim_g
        fprintf('Constraint computation time: %.2fms\n', 1000*time(2)/size(hive, 1))
        fprintf('Lagrangian computation time: %.2fms\n', 1000*time(3)/size(hive, 1))
        fprintf('Estimated tot. comp. time:   %.0fm %.0fs\n',...
                    floor((time(1) + time(2))*cycle/60), mod((time(1) + time(2))*cycle, 60))
    else
        fprintf('Cost fun. computation time:  %.2fms\n', 1000*time(1)/size(hive, 1))
        fprintf('Estimated tot. comp. time:   %.0fm %.0fs\n',...
                    floor(time(1)*cycle/60), mod(time(1)*cycle, 60))
    end
    reply = input('Do you want to continue? Y/N [Y]: ', 's');
    switch reply
        case 'N', ABC_time = 0; opt = hive(end-n_opt+1:end, :); return
        otherwise, clear reply time
    end
end

%% Algorithm
lenDisp = 0; tic
for iter = 1:cycle
    if opts.v
        fprintf([repmat('\b', 1, lenDisp)]);
        lenDisp = fprintf('Iter: %d of %d\n', iter, cycle);
    end

    % Resample onlooker bees
    if ~dim_g, fit = abs(f_f(1:n_emp, :) - f_f(end-n_opt+1, :));
    else, fit = abs(f_dL(1:n_emp, :) - f_dL(end-n_opt+1, :)); end
    fit = max(fit) - fit;

    % Probability and density function
    if sum(fit) < 1e-5, prob = ones(n_emp, 1)/n_emp;
    else, prob = fit/sum(fit); end
    dens_prob = cumsum(prob);

    for i = 1:n_onl
        index = find(dens_prob >= rand, 1);
        hive(n_emp + i, :) = hive(index, :);
        if ~dim_g, f_f(n_emp + i, :) = f_f(index, :);
        else, f_g(n_emp + i, :) = f_g(index, :); f_dL(n_emp + i, :) = f_dL(index, :); end
    end

    % Move bees
    for i = 1:n_emp + n_onl
        % Choose bee to comapre
        k = i;
        while k == i
            k = randi(n_emp + n_onl, 1);
            j = randperm(dim + dim_g, randi(dim + dim_g, 1));
        end
        
        % Optimize solution
        new_sol = hive(i, :);
        new_sol(1, j) = new_sol(1, j) + phi(length(j)).*(new_sol(1, j) - hive(k, j));
        
        if ~dim_g
            f_new = f(new_sol);
            [hive(i, :), check] = feasPos(new_sol(1, 1:dim), hive(i, 1:dim), ...
                f_new, f_f(i, :), lb, ub);
            if check, f_f(i, :) = f_new; end
        else
            g_new = g(new_sol(1:dim));
            dL_new = sum(discDiff(L, new_sol, dim).^2) + sum(g_new.^2);
            [~, check] = feasPos(new_sol(1, 1:dim), hive(i, 1:dim), ...
                dL_new, f_dL(i, :), lb, ub);
            if check, hive(i, :) = new_sol; f_g(i, :) = g_new; f_dL(i, :) = dL_new; end
        end

        if check, n_nup(i) = 0; else, n_nup(i) = n_nup(i) + 1; end
    end
    

    % Optimal solution
    if ~dim_g
        [hive(end-n_opt+1:end, :), best_index] = optSol(f_f, hive, dim, n_opt, tol);
        f_f(end-n_opt+1:end, :) = f_f(best_index, :);
    else
        [hive(end-n_opt+1:end, :), best_index] = optSol(f_dL, hive, dim, n_opt, tol);
        f_g(end-n_opt+1:end, :) = f_g(best_index, :);
        f_dL(end-n_opt+1:end, :) = f_dL(best_index, :);
    end

    % Plot hive
    if opts.showFig(2)
        if ~dim_g, drawHiveFun(opts.nFig, dim, hive, f, f_f, n_emp, n_onl, ...
                        n_opt, lb, ub);
        else, drawHiveLag(opts.nFig, dim, hive, g, L, f_dL, n_emp, n_onl, n_opt, lb, ub)
        end
    end

    % Reject non updated bees
    for i = 1:n_emp
        if n_nup(i) >= maxIter
            hive(i, :) = gen(1); n_nup(i, :) = 0;
            if ~dim_g, f_f(i, :) = f(hive(i, 1:dim));
            else
                f_g(i, :) = g(hive(i, 1:dim));
                f_dL(i, :) = sum(discDiff(L, hive(i, :), dim).^2) + sum(f_g(i, :).^2);
            end
        end
    end
end

%% Results
ABC_time = toc;
opt = hive(end-n_opt+1:end, :);
if opts.showFig(3)
    if ~dim_g, drawHiveFun(opts.nFig, dim, hive, f, f_f, n_emp, n_onl, ...
                    n_opt, lb, ub);
    else, drawHiveLag(opts.nFig, dim, hive, g, L, f_dL, n_emp, n_onl, n_opt, lb, ub)
    end
end
end

%% Functions
function dfun = discDiff(fun, x, dim)
    % Return the discretized differentiation of the function
    % Input:
    %   fun     - function [@fun(), mex()]
    %   x       - point where to evalute the derivate   [doubel(1, n)]
    % Output:
    %   dfun    - disc diff of the funtion              [double(n, length(fun))] 
    
    alpha = 1e-5; % differentaition step
    alpha_x = alpha*eye(dim, length(x));
    dfun = zeros(dim, length(fun(x)));
    for i = 1:dim
        dfun(i, :) = (fun(x + alpha_x(i, :)) - fun(x - alpha_x(i, :))) / alpha;
    end
end

function [opt, best_index] = optSol(cost, hive, dim, n_opt, tol)
    % Return n_opt indeces of the bees  with minimum cost, distant each
    % other more than tol (w.r.t. the first dim dimensions)
    % Input:
    %   cost    - cost of each bee              [double(n, 1)]
    %   hive    - bees positions                [double(n, m) m >= dim]
    %   dim     - problem's dimension           [-]
    %   n_opt   - number of optimal solutions   [-]
    %   tol     - min dist between opt sols     [-]
    % Output:
    %   opt         - cost of the optimal solutions     [double(n_opt, 1)]
    %   best_index  - index of the best solutions       [double(n_opt, 1)]

    [~, index] = sort(cost);
    best_index = index(1:n_opt); opt = hive(index(1:n_opt), :);
    if n_opt == 1, return, end
    iter = 2;
    for i = 2:size(hive, 1)
        if all(sqrt(sum((opt(1:iter-1, 1:dim) - hive(index(i), 1:dim)).^2, 2)) > tol)
            best_index(iter) = index(i);
            opt(iter, :) = hive(index(i), :);
            iter = iter + 1;
        end

        if iter > n_opt, return, end
    end
end

function [p, check] = feasPos(p1, p2, f1, f2, lb, ub)
     % Return the position s.t.
    %   - if p1, p2 unfeasible  -> p = the unfeasible p
    %       (euclidian distance from the bounds)
    %   - if one is unfeasible  -> p = feasible one
    %   - if both feasible      -> p = p with min f_p
    %
    % Input:
    %   p1      - position 1           [double(1, dim)]
    %   p2      - position 2           [double(1, dim)]
    %   f1      - cost p1              [-]
    %   f2      - cost p2              [-]
    %   lb      - lower bound          [double(1, dim)]
    %   ub      - upper bound          [double(1, dim)]
    %
    % Output:
    %   p       - best position        [double(1, dim)]
    %   check   - true if p == p1      [boolean]

    p = [p1; p2]; check = false;

    % Compute feasibility of the solution
    feas_lb = p >= lb; feas_ub = p <= ub;
    feas = all(feas_lb & feas_ub, 2);

    % Compute distance from bounds
    pos_ub = p - ub; pos_lb = p - lb;
    pos_ub(feas_ub) = 0; pos_lb(feas_lb) = 0;
    d = sum(pos_ub.^2, 2) + sum(pos_lb.^2, 2);

    switch feas(1)*2 + feas(2)
        case 0 % Both unfeasible
            if d(1) < d(2), p = p1; check = true;
            else, p = p2; end
        case 3
            if f1 <= f2, p = p1; check = true;
            else, p = p2; end
        otherwise
            p = p(feas, :); check = feas(1);
    end
end

function drawHiveFun(nFig, dim, hive, f, f_f, n_emp, n_onl, n_opt, lb, ub)
    % Plot the 2D or 3D graph of the system
    % Input:
    %   nFig    - figure #                      [-]
    %   dim     - problem's dimension           [-]
    %   hive    - possible solutions            [double(n, m)]
    %   f       - (nectar) cost function        [@fun(), mex()]
    %   f_f     - hive cost                     [double(n, 1)]
    %   n_emp   - # of employed bees            [-]
    %   n_onl   - # of onlooker bees            [-]
    %   n_opt   - # of optimal bees             [-]
    %   lb      - system lower bounds           [double(1, dim)]
    %   ub      - system upper bounds           [double(1, dim)]
    
    figure(nFig), hold off
    
    if dim < 2
        % Scalar case
        x = lb:(ub-lb)/100:ub; f_fplot = zeros(length(x), 1);
        for i = 1:length(x), f_fplot(i, :) = f(x(i)); end

        plot(x, f_fplot, 'LineWidth', 1), hold on
        scatter(hive(1:n_emp, 1), f_f(1:n_emp, :), 'b', 'filled')
        scatter(hive(n_emp+1:n_emp+n_onl, 1), f_f(n_emp+1:n_emp+n_onl, :), 'r', 'filled')
        scatter(hive(end-n_opt+1:end, 1), f_f(end-n_opt+1:end, :), 'g', 'filled')
        xlabel('sol'), ylabel('cost fun'), grid on; drawnow
    end

    if dim >= 2
        % Multiple dimension
        [x, y] = meshgrid(lb(1):(ub(1)-lb(1))/100:ub(1), lb(2):(ub(2)-lb(2))/100:ub(2));
        f_fplot = zeros(size(x));
        for i = 1:length(x)
            for j = 1:length(x)
                f_fplot(i, j) = f([x(i, j), y(i, j), hive(end-n_opt+1, 3:dim)]);
            end
        end

        surf(x, y, f_fplot), hold on
        scatter3(hive(1:n_emp, 1), hive(1:n_emp, 2), f_f(1:n_emp, :), 'b', 'filled')
        scatter3(hive(n_emp+1:n_emp+n_onl, 1), hive(n_emp+1:n_emp+n_onl, 2),...
            f_f(n_emp+1:n_emp+n_onl, :), 'r', 'filled')
        scatter3(hive(end-n_opt+1:end, 1), hive(end-n_opt+1:end, 2),...
            f_f(end-n_opt+1:end, :), 'g', 'filled')
        xlabel('x sol'), ylabel('y sol'), zlabel('cost fun')
        grid on; drawnow
    end
end

function drawHiveLag(nFig, dim, hive, g, L, f_dL, n_emp, n_onl, n_opt, lb, ub)
    % Plot the 2D or 3D graph of the system
    % Input:
    %   nFig    - figure #                      [-]
    %   dim     - problem's dimension           [-]
    %   hive    - possible solutions            [double(n, m)]
    %   g       - equality constraint           [@fun(), mex()]
    %   L       - Lagrangian equation           [@fun(), mex()]
    %   f_dL    - Lag diff. cost                [double(n, 1)]
    %   n_emp   - # of employed bees            [-]
    %   n_onl   - # of onlooker bees            [-]
    %   n_opt   - # of optimal bees             [-]
    %   lb      - system lower bounds           [double(1, dim)]
    %   ub      - system upper bounds           [double(1, dim)]
    
    figure(nFig), hold off
    
    if dim == 1
        lb = [lb, min(hive(:, dim+1:end), [], 'all')];
        ub = [ub, max(hive(:, dim+1:end), [], 'all')];
    end

    % Multiple dimension
    [x, y] = meshgrid(lb(1):(ub(1)-lb(1))/100:ub(1), lb(2):(ub(2)-lb(2))/100:ub(2));
    f_dLplot = zeros(size(x));
    for i = 1:length(x)
        for j = 1:length(x)
            point = [x(i, j), y(i, j), hive(end-n_opt+1, 3:end)];
            f_dLplot(i, j) = sum(discDiff(L, point, dim).^2) + sum(g(point(1:dim)).^2);
        end
    end

    surf(x, y, log(f_dLplot)), hold on
    scatter3(hive(1:n_emp, 1), hive(1:n_emp, 2), log(f_dL(1:n_emp, :)), 'b', 'filled')
    scatter3(hive(n_emp+1:n_emp+n_onl, 1), hive(n_emp+1:n_emp+n_onl, 2),...
        log(f_dL(n_emp+1:n_emp+n_onl, :)), 'r', 'filled')
    scatter3(hive(end-n_opt+1:end, 1), hive(end-n_opt+1:end, 2),...
        log(f_dL(end-n_opt+1:end, :)), 'g', 'filled')
    xlabel('x sol'), ylabel('y sol'), zlabel('cost fun')
    grid on; drawnow
end