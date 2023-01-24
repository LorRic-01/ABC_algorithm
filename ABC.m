function [opt, hive, ABC_time] = ABC(dim, f, lb, ub, g, type, ...
                n_emp, n_onl, cycle, gen, phi, maxIter, ...
                nEqLag, n_opt, tol, hive_i)
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
%      n_emp    - # of employed bees                            [100 (default)]
%      n_onl    - # of onlooker bees                            [100 (default)]
%      gen      - bees initialization function                  [1000*rand - 500 (default) | @fun(), mex()]
%      phi      - random function in [-1, 1]                    [2*rand - 1 (default) | @fun(), mex()]
%      maxIter  - max non-update sol iter before rejection      [10 (default)]
%      nEqLag   - max # of equation to compute Lag multiplier   [dim (default)]
%      hive_i   - hive initialization                           [double(n_i, dim + dim_g)]
%     
%   Solution settings
%      n_opt  - number of returned optimal sol          [10 (default)]
%      tol    - tolerance to discard returned near sol  [1 (default)]
%
%   Output:
%      opt      - optimal solutions   [double(n_opt, dim)]
%      hive     - general hive        [double(n_emp + n_onl + n_opt, dim + dim_g)]
%      ABC_time - sol. comp. time     [double]

%% internal parameters
showFig = [false, false, true];     % show figure ([init, loop, end])
nFig = 1;                           % figure # for plotting

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
if (nargin < 13) || isempty(nEqLag),  nEqLag = dim; end
if (nargin < 14) || isempty(n_opt),   n_opt = 10; end
if (nargin < 15) || isempty(tol),     tol = 1; end

% Additional parameters
if ~isempty(g)
    dim_g = length(g(zeros(1, dim)));
    if size(g(zeros(1, dim)), 2) ~= dim_g
        g = @(x) g(x)';
    end
else
    g = @(x) 0; dim_g = 0;
end

switch type
    case 'max', type = -1;
    otherwise,  type = 1;
end

%% Hive initialization
if (nargin < 16) || isempty(hive_i) || size(hive_i, 2) ~= dim + dim_g
    hive = gen(n_emp + n_onl + n_opt, dim + dim_g);
else
    hive = [hive_i(1:min(size(hive_i, 1), n_emp + n_onl + n_opt), :);
    gen(min(n_emp + n_onl + n_opt - size(hive_i, 1), 0), dim + dim_g)];
end

% Cost function values allocation
f_f = zeros(size(hive, 1), 1); f_g = zeros(size(hive, 1), max(dim_g, 1));
f_L = zeros(size(hive, 1), 1);
time = [0, 0, 0];
for i = 1:size(hive, 1)
    tic, f_f(i, :) = f(hive(i, 1:dim)); time(1) = time(1) + toc;
    tic, f_g(i, :) = g(hive(i, 1:dim)); time(2) = time(2) + toc;
    tic, f_L(i, :) = f_f(i, :) + sum(hive(i, dim+1:end).*f_g(i, :)); time(3) = time(3) + toc;
end
fprintf('Cost fun. computation time:  %.2fms\n', 1000*time(1)/size(hive, 1))
fprintf('Constraint computation time: %.2fms\n', 1000*time(2)/size(hive, 1))
fprintf('Lagrangian computation time: %.2fms\n', 1000*time(3)/size(hive, 1))
fprintf('Estimated tot. comp. time:   %.0fm %.0fs\n',...
    floor((sum(time) + (time(1) + time(2))*min(nEqLag, dim))*cycle/60),...
    mod((sum(time) + (time(1) + time(2))*min(nEqLag, dim))*cycle, 60))
n_nup = zeros(size(hive, 1), 1);        % # of non updated iteration

% Optimal solutions
hive(end-n_opt+1:end, :) = optSol(dim, hive, type, f_f, sum(f_g, 2), n_opt, tol);
for i = size(hive, 1) - n_opt + 1:size(hive, 1)
    f_f(i, :) = f(hive(i, 1:dim)); f_g(i, :) = g(hive(i, 1:dim));
    f_L(i, :) = f_f(i, :) + sum(hive(i, dim+1:end).*f_g(i, :));
end

if showFig(1), drawHive(nFig, dim, hive, f, g, f_L, n_emp, n_onl, n_opt, [min(lb), max(ub)]); end

reply = input('Do you want to continue? Y/N [Y]: ', 's');
switch reply
    case 'N', opt = nan(1, dim); hive = nan(1, dim + dim_g); ABC_time = 0; return
    otherwise, clear reply time
end

%% Algorithm
tic
for iter = 1:cycle
    % Resample onlooker bees
    if type == -1, score = abs(max(f_L(1:n_emp)) - f_L(1:n_emp)) + sum(abs(f_g(1:n_emp, :)), 2);
    else, score = abs(min(f_L(1:n_emp)) - f_L(1:n_emp)) + sum(abs(f_g(1:n_emp, :)), 2); end

    % Probability and its density function
    prob = (max(score) - score)/sum(max(score) - score);
    dens_prob = cumsum(prob);

    for k = 1:n_onl
        index = find(dens_prob >= rand, 1);
        hive(n_onl + k, :) = hive(index, :);
    end
    
    hive_i = hive; % hive int he current iteration
    % Move bees
    for i = 1:n_emp + n_onl
        % Choose bee to compare and in which dimension
        k = i;
        while k == i
            k = randi(n_emp + n_onl, 1);
            j = randi(dim + dim_g, 1);
        end

        if j <= dim
            % Solution optimization
            new_sol = hive_i(i, :);
            new_sol(1, j) = new_sol(1, j) + phi()*(new_sol(1, j) - hive_i(k, j));
            f_new = f(new_sol(1, 1:dim)); g_new = g(new_sol(1, 1:dim));
            L_new = f_new + sum(new_sol(1, dim+1:end).*g_new);

            % Move if it is convenient
            [hive(i, 1:dim), check] = feasPos(new_sol(1, 1:dim), hive(i, 1:dim), ...
                                type*L_new, type*f_L(i, :), lb, ub);
            if check, f_f(i, :) = f_new; f_g(i, :) = g_new; f_L(i, :) = L_new; end
        else
            % Lagrangian multipliers optimization
            % Mve toward the solution
            j1 = randi(dim, 1);
            new_sol = hive_i(i, :);
            new_sol(1, j1) = new_sol(1, j1) + phi()*(new_sol(1, j1) - hive_i(k, j1));
            f_new = f(new_sol(1, 1:dim)); g_new = g(new_sol(1, 1:dim));
            L_new = f_new + sum(new_sol(1, dim+1:end).*g_new);

            % Move if it is convenient
            [hive(i, 1:dim), check] = feasPos(new_sol(1, 1:dim), hive(i, 1:dim), ...
                                sum(abs(g_new), 2), sum(abs(f_g(i, :)), 2), lb, ub);
            if check, f_f(i, :) = f_new; f_g(i, :) = g_new; f_L(i, :) = L_new; end

            % Update lagrangian multipliers
            alpha = min([0.01, abs(f_g(i, :))]);
            df = zeros(min(dim, nEqLag), 1); dg = zeros(min(dim, nEqLag), dim_g);
            dx = diag(alpha*hive(i, 1:dim)); hive(i, j) = inf;
            while isinf(hive(i, j)) || isnan(hive(i, j))
                indexLag = [j1, randi(dim, 1, min(dim, nEqLag) - 1)];
                for n = 1:min(dim, nEqLag)
                    df(n, :) = f(hive(i, 1:dim) + dx(indexLag(n), :)) - ...
                        f(hive(i, 1:dim) - dx(indexLag(n), :));
                    dg(n, :) = g(hive(i, 1:dim) + dx(indexLag(n), :)) - ...
                        g(hive(i, 1:dim) - dx(indexLag(n), :));
                end

                hive(i, j) = -dg(:, j-dim)'*df/(norm(dg(:, j-dim)).^2);
                f_L(i, :) = f_f(i, :) + hive(i, dim+1:end)*f_g(i, :)';
            end
        end

        if check, n_nup(i) = n_nup(i) + 1; end
    end
    
    % Optimal solution
    hive(end-n_opt+1:end, :) = optSol(dim, hive, type, f_f, sum(f_g, 2), n_opt, tol);
    for i = size(hive, 1) - n_opt + 1:size(hive, 1)
        f_f(i, :) = f(hive(i, 1:dim)); f_g(i, :) = g(hive(i, 1:dim));
        f_L(i, :) = f_f(i, :) + sum(hive(i, dim+1:end).*f_g(i, :));
    end

    if showFig(2), drawHive(nFig, dim, hive, f, g, f_L, n_emp, n_onl, n_opt, [min(lb), max(ub)]); end

    % Resample non updted iterations
    hive(n_nup >= maxIter, :) = gen(sum(n_nup >= maxIter), dim + dim_g);
    n_nup(n_nup >= maxIter, :) = zeros(sum(n_nup >= maxIter), 1);
end

opt = hive(end-n_opt+1:end, 1:dim);
ABC_time = toc;
if showFig(3), drawHive(nFig, dim, hive, f, g, f_L, n_emp, n_onl, n_opt, [min(lb), max(ub)]); end

end

%% Functions
function opt = optSol(dim, hive, type, f_f, f_g, n_opt, tol)
    % Return the first n_opt optimal points of the current hive s.t. are
    % the nearest w.r.t. the constraint and best cost
    % Input:
    %   dim    - problem's dimension        [-]
    %   hive   - general hive               [double(n_emp + n_onl + n_opt, dim + dim_g)]
    %   type   - optimization type          [-1 -> maximization, 1 -> minimization]
    %   f_f    - (nectar) cost function     [double(n_emp + n_onl + n_opt, 1)]
    %   f_g    - equality constraint        [double(n_emp + n_onl + n_opt, dim_g)]
    %   n_opt  - # of returned optimal sol  [-]
    %   tol    - tolerance to discard       [-]
    % Output:
    %   opt - optimal solutions             [double(n_opt, dim + dim_g)]

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

function drawHive(nFig, dim, hive, f, g, f_L, n_emp, n_onl, n_opt, minmax)
    % Plot the 2D or 3D graph of the system
    % Input:
    %   nFig    - figure #                      [-]
    %   dim     - problem's dimension           [-]
    %   hive    - possible solutions            [double(n, dim + dim_g)]
    %   f       - (nectar) cost function        [@fun(), mex()]
    %   g       - equality constraint           [none (default) | @fun(), mex()]
    %   f_L     - hive Lagrangian cost          [double(n, 1)]
    %   n_emp   - # of employed bees            [-]
    %   n_onl   - # of onlooker bees            [-]
    %   n_opt   - # of returned optimal sol     [-]
    %   minmax  - plot bounds                   [double(1, 2)]
    
    warning off
    
    L = @(x) f(x(:, 1:dim)) + sum(x(:, dim+1:size(hive, 2)).*g(x(:, 1:dim)), 2); % Lagrangian
    
    figure(nFig), hold off
    if size(hive, 2) < 2
        fplot(@(x) L(x), minmax), hold on
        scatter(hive(1:n_emp, 1), f_L(1:n_emp, :), 'r', 'filled')
        scatter(hive(n_emp+1:n_emp+n_onl, 1), f_L(n_emp+1:n_emp+n_onl, :), 'b', 'filled')
        scatter(hive((end - n_opt + 1):end, 1), f_L((end - n_opt + 1):end, :), 'g', 'filled')
        xlabel('x'), ylabel('L(x)'), drawnow
    else
        fsurf(@(x, y) L([x, y, hive(end-n_opt+1, 3:end)]), minmax), hold on
        scatter3(hive(1:n_emp, 1), hive(1:n_emp, 2), f_L(1:n_emp, :), 'r', 'filled')
        scatter3(hive(n_emp+1:n_emp+n_onl, 1), hive(n_emp+1:n_emp+n_onl, 2), f_L(n_emp+1:n_emp+n_onl, :), 'b', 'filled')
        scatter3(hive((end - n_opt + 1):end, 1), hive((end - n_opt + 1):end, 2), f_L((end - n_opt + 1):end, :), 'g', 'filled')
        xlabel('x1'), ylabel('x2'), zlabel('L'), drawnow
    end
    
    warning on
end

function [p, f_p, chech] = feasPos(p1, p2, f_p1, f_p2, lb, ub)
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
    
    p = [p1; p2]; f_p = [f_p1; f_p2]; chech = false;

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
            if d(1) > d(2), p = p2; f_p = f_p2; chech = true;
            else, p = p1; f_p = f_p1; end
        case 3
            if f_p1 < f_p2, p = p1; f_p = f_p1;
            else, p = p2; f_p = f_p2; chech = true; end
        otherwise
            p = p(feas, :); f_p = f_p(feas, :);
            chech = feas(2);
    end
end