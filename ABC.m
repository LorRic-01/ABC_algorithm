function [opt, ABC_time, hive] = ABC(dim, f, lb, ub, g, ...
                       n_emp, n_onl, gen, phi, maxIter, ...
                       n_opt, tol, opts, typeVal, hive_i)
% ABC - Artificial Bee Colony algorithm with Augmented Lagrngian method for
%       Equality Constrained optimization
% Optimization algorithm that solves min problems in the form
%   argmin f(x)     subject to:        g(x) = 0       (equality constraints)
%     x                             lb <= x <= ub     (bounds)
%
% Problem settings:
%   dim  - problem's dimension     [-]
%   f    - (nectar) cost function  [@fun(), mex]
%   g    - equality constraints    [empty (default) | @fun(), mex]
%   lb   - solution lower bound    [-inf (default) | double(1, dim)]
%   ub   - solution upper bound    [inf (default) | double(1, dim)]
%
% Colony settings:
%   n_emp    - # of employed bees                           [100 (default)]
%   n_onl    - # of onlooker bees                           [100 (default)]
%   gen      - bees inizialization function                 [@(n) 200*rand(n , dim) - 100 (default) | @fun(), mex]
%   phi      - random function between [-1, 1]              [@(n) 2*rand(n, 1) - 1 (default) | @fun(), mex]
%   maxIter  - max non-updated sol. iter before rejection   [25 (default)]
%   hive_i   - hive initialization                          [double(n_i, dim + dim_g)]
%
% Result settings:
%   n_opt  - number of optimal solutions          [10 (default)]
%   tol    - min distance between opt solutions   [1 (default)]
%
% Options:
%   opts     - ABC method options                               [struct]
%                'nFig'     - # of figure for plotting          [1 (default)]
%                'showFig'  - show figure ([init, loop, end])   [[false, false, false] (default)]
%                'v'        - verbose                           [false (default)]
%                'type'     - optimization cycle type           ['iter' (default) | 'step']
%                'single'   - function defined for single input [false (default)]
%
%   typeVal  - optimization cycle type value                    [100 (default)]
%
% Output:
%   opt       - optimal solutions       [double(n_opt, dim)]
%   ABC_time  - sol. computation time   [double]
%   hive      - hive                    [double(n_emp + b_emp, n_opt, dim + dim_g)]

%% Default parameters
if nargin < 2 || isempty(dim) || isempty(f)
    error('Missing problem dimension and/or cost function')
end
if nargin <  3 || isempty(lb),      lb = -inf(1, dim);                  end
if nargin <  4 || isempty(ub),      ub =  inf(1, dim);                  end
if nargin <  5 || isempty(g),       g = [];                             end
if nargin <  6 || isempty(n_emp),   n_emp = 100;                        end
if nargin <  7 || isempty(n_onl),   n_onl = 100;                        end
if nargin <  8 || isempty(gen),     gen = @(n) 200*rand(n , dim) - 100; end
if nargin <  9 || isempty(phi),     phi = @(n) 2*rand(n, 1) - 1;        end
if nargin < 10 || isempty(maxIter), maxIter = 25;                       end
if nargin < 11 || isempty(n_opt),   n_opt = 10;                         end
if nargin < 12 || isempty(tol),     tol = 1;                            end
if nargin < 13 || isempty(opts)
    opts = struct('nFig', 1, 'showFig', [false, false, false], ...
        'v', false, 'type', 'iter', 'single', false);
end
if nargin < 14 || isempty(typeVal), typeVal = 100;                      end

switch opts.type
    case 'step', opts.type = false;
    otherwise, opts.type = true;
end

%% Additional checks and parameters
% Equality cosntraints
if nargin <  5 || isempty(g), g = []; dim_g = 0;
else
    dim_g = length(g(zeros(1, dim)));
    if size(g(zeros(1, dim)), 2) ~= dim_g, g = @(x) g(x)'; end
    gen = @(n) [gen(n), 10*randn(n, dim_g) - 5];
end

% Function type check
handle = isa(f, 'function_handle') && ...
            (isempty(g) || isa(g, 'function_handle')) && ~opts.single;
if ~handle
    f = @(x) fun_eval(f, x);
    if dim_g, g = @(x) fun_eval(g, x); end
end

% Lagrangian function creation
if dim_g
    L = @(x) f(x(:, 1:dim)) + sum(x(:, dim+1:end).*g(x(:, 1:dim)), 2);
    f = @(x) sum(dfun_eval(L, x).^2, 2);
end

%% Hive initialization
if nargin < 15 || isempty(hive_i) || size(hive_i, 2) ~= dim + dim_g
    hive = gen(n_emp + n_onl + n_opt);
else
    n_bees = n_emp + n_onl + n_opt;
    hive = [hive_i(1:min(n_bees, size(hive_i, 1)), :); ...
        gen(max(n_bees - size(hive_i, 1), 0))];
end
n_nup = zeros(size(hive, 1), 1);

% Cost function evaluation
tic, cost = f(hive); time = toc;

% Optimal solution
[hive(end-n_opt+1:end, :), index] = optSol(cost, hive, dim, n_opt, tol);
cost(end-n_opt+1:end, :) = cost(index, :);

%% Plot and verbose
if opts.showFig(1)
    drawHive(opts.nFig, dim, hive, f, cost, n_emp, n_onl, n_opt, lb, ub)
end

if opts.v
    if opts.type
        fprintf('Cost function computation time: %.2fms\n', 1000*time/size(hive, 1));
        fprintf('Estimated tot computation time: %.0fm %.0fs\n', ...
            floor(time*typeVal/60), mod(time*typeVal, 60));
    end

    reply = input('Do you want to continue Y/N [Y]: ', 's');
    switch reply
        case 'N', ABC_time = 0; opt = hive(end-n_opt+1:end, 1:dim); return
        otherwise, clear reply
    end
end

%% Algorithm
lenDisp = 0; tic
if opts.type, cycle = typeVal; else, cycle = inf; end
for iter = 1:cycle
    if opts.v && opts.type
        fprintf([repmat('\b', 1, lenDisp)]);
        lenDisp = fprintf('Iter: %d of %d\n', iter, cycle);
    end
    
    % Resample onlooker bees
    fit = abs(cost(1:n_emp) - cost(end-n_opt+1));
    fit = max(fit) - fit;

    % Probability and dednsity function
    if sum(fit) < 1e-5, prob = ones(n_emp, 1)/n_emp;
    else, prob = fit/sum(fit); end
    dens_prob = cumsum(prob);

    for i = 1:n_onl
        index = find(dens_prob >= rand, 1);
        hive(n_emp + i, :) = hive(index, :);
        cost(n_emp + i) = cost(index);
    end

    % Move bees
    tmp_hive = hive;
    for i = 1:n_emp + n_onl
        k = i;
        while k == i
            k = randi(n_emp + n_onl, 1);
            j = randperm(dim + dim_g, randi(dim + dim_g , 1)); % randi(dim + dim_g, 1);
        end
        tmp_hive(i, j) = tmp_hive(i, j) + phi(1).*(tmp_hive(i, j) - hive(k, j));
    end
    new_cost = f(tmp_hive);

    % Best poition
    for i = 1:n_emp + n_onl
        check = bestPos(hive(i, 1:dim), tmp_hive(i, 1:dim), ...
            cost(i), new_cost(i), lb, ub);
        if ~check, hive(i, :) = tmp_hive(i, :); cost(i) = new_cost(i); n_nup(i) = 0;
        else, n_nup(i) = n_nup(i) + 1; end
    end

    % Optimal solution
    [hive(end-n_opt+1:end, :), index] = optSol(cost, hive, dim, n_opt, tol);
    cost(end-n_opt+1:end, :) = cost(index, :);

    % Plot hive
    if opts.showFig(2)
        drawHive(opts.nFig, dim, hive, f, cost, n_emp, n_onl, n_opt, lb, ub)
    end

    % Reejct non updated bees
    hive(n_nup >= maxIter, :) = gen(sum(n_nup >= maxIter));
    cost(n_nup >= maxIter) = f(hive(n_nup >= maxIter, :));
    n_nup(n_nup >= maxIter) = zeros(sum(n_nup >= maxIter), 1);
end

%% Results
ABC_time = toc;
opt = hive(end-n_opt+1:end, 1:dim);

% Plot hive
if opts.showFig(3)
    drawHive(opts.nFig, dim, hive, f, cost, n_emp, n_onl, n_opt, lb, ub)
end

end

%% Function
function f_data = fun_eval(f, data)
    % Evalute the function in the data
    % Input:
    %   f       - function to evaluate [@fun(), mex (return double(:, k))]
    %   data    - data to use [double(n, m)]
    % Output
    %   f_data  - f(data) [double(n, k)]

    f_data = zeros(size(data, 1), length(f(data(1,:))));
    for i = 1:size(data, 1)
        f_data(i, :) = f(data(i, :));
    end
end

function df_data = dfun_eval(f, data)
    % Compute the numeric derivate of the function in the data
    % Input:
    %   f       - function to evaluate [@fun(), mex (return double(:, 1))]
    %   data    - data to use [double(n, m)]
    % Output
    %   df_data  - df(data) [double(n, m)]

    df_data = zeros(size(data)); alpha = 1e-5; % differentiation step
    alpha_x = alpha*eye(size(data, 2));
    for i = 1:size(data, 1)
        df_data(i, :) = (f(data(i, :) + alpha_x) - f(data(i, :) - alpha_x))/alpha;
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

function drawHive(nFig, dim, hive, f, cost, n_emp, n_onl, n_opt, lb, ub)
    % Plot the 2D or 3D graph of the system
    % Input:
    %   nFig    - figure #                      [-]
    %   dim     - problem's dimension           [-]
    %   hive    - possible solutions            [double(n, m)]
    %   f       - cost function                 [@fun(), mex()]
    %   cost    - hive cost                     [double(n, 1)]
    %   n_emp   - # of employed bees            [-]
    %   n_onl   - # of onlooker bees            [-]
    %   n_opt   - # of optimal bees             [-]
    %   lb      - system lower bounds           [double(1, dim)]
    %   ub      - system upper bounds           [double(1, dim)]
    
    figure(nFig), hold off
    
    n = [n_emp, n_onl, n_opt]; n = cumsum(n);
    lb = max(lb, -100); ub = min(ub, 100);

    if size(hive, 2) < 2
        % Scalar case
        x = lb:(ub - lb)/100:ub; cost_plot = f(x');
        scatter(hive(1:n(1), 1), cost(1:n(1)), 'b', 'filled'), hold on
        scatter(hive(n(1)+1:n(2), 1), cost(n(1)+1:n(2)), 'r', 'filled')
        scatter(hive(n(2)+1:n(3), 1), cost(n(2)+1:n(3)), 'g', 'filled')
        plot(x, cost_plot, 'LineWidth', 1)
        xlabel('sol'), ylabel('Cost'), legend('emp', 'onl', 'opt'), grid on
        axis tight; drawnow
    else
        % 3D case
        if dim < 2
            lb = [lb, min(hive(:, dim+1:end), [], 'all')];
            ub = [ub, max(hive(:, dim+1:end), [], 'all')];
        end
        [x, y] = meshgrid(lb(1):(ub(1) - lb(1))/100:ub(1), lb(2):(ub(2) - lb(2))/100:ub(2));
        x = reshape(x, [], 1); y = reshape(y, [], 1);
        cost_plot = f([x, y, ones(size(x, 1), 1)*hive(n(2)+1, 3:end)]);
        x = reshape(x, 101, 101); y = reshape(y, 101, 101); cost_plot = reshape(cost_plot, 101, 101);
        scatter3(hive(1:n(1), 1), hive(1:n(1), 2), cost(1:n(1)), 'b', 'filled'), hold on
        scatter3(hive(n(1)+1:n(2), 1), hive(n(1)+1:n(2), 2), cost(n(1)+1:n(2)), 'r', 'filled')
        scatter3(hive(n(2)+1:n(3), 1), hive(n(2)+1:n(3), 2), cost(n(2)+1:n(3)), 'g', 'filled')
        surf(x, y, cost_plot)
        xlabel('sol1'), ylabel('sol2'), zlabel('Cost'), legend('emp', 'onl', 'opt'), grid on
        axis tight; drawnow
    end
end

function check = bestPos(p1, p2, f1, f2, lb, ub)
    % Return true if
    %   - p1, p2 unfeasible BUT p1 less unfeasible than p2
    %   - p1 feasible, p2 unfeasible
    %   - both feasible BUT f1 < f2
    %
    % Input:
    %   p1  - position 1    [double(1, dim)]
    %   p2  - position 2    [double(1, dim)]
    %   f1  - cost p1       [-]
    %   f2  - cost p2       [-]
    %   lb  - lower bound   [double(1, dim)]
    %   ub  - upper bound   [double(1, dim)]
    %
    % Output:
    %   check               [boolean]

    p = [p1; p2];

    % Feasibility
    feas = all((p >= lb) & (p <= ub), 2);

    % Distance from bounds
    pos_ub = p - ub; pos_lb = lb - p;
    pos_ub(pos_ub > 0) = 0; pos_lb(pos_lb > 0) = 0;
    d = sum(pos_ub.^2, 2) + sum(pos_lb.^2, 2);

    switch feas(1)*2 + feas(2)
        case 0, check = d(1) < d(2); % both unfeasible
        case 3, check = f1 < f2;
        otherwise, check = feas(1);
    end
end