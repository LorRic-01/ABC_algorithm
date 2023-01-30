function [opt, hive, ABC_time] = ABC(dim, f, lb, ub, ...
                n_emp, n_onl, cycle, gen, phi, maxIter,...
                g, lamLimit, lagCycle, ...
                opts, opt_i)
% ABC - Artificial Bee Colony algorithm with Augmented Lagrangian Method for
% Equality-Constrained Optimization
%   Optimization algorithm that solves mir or max problems in the form
%   arg__ f(x)  subject to:     g(x) = 0    (equality constraints)
%       x                    lb <= x <= ub  (bounds)
%
%   Problem settings:
%      dim      - problem's dimension         [-]
%      f        - (nectar) cost function      [@fun(), mex()]
%      lb       - solution lower bound        [-inf (default) | double(1, dim)]
%      ub       - solution upper bound        [ inf (default) | double(1, dim)]
%      cycle    - # of algorithm iterations   [100 (default)]
%
%   Problem constraints settings:
%      g        - equality constraints         [none (default) | @fun(), mex()]
%      lamLimit - Lagrange multipliers limit   [[-1e10, 1e10] (default) | double(1, 2)]
%      lagCycle - updating cycle of Lag mult   [10 (default)]
%
%   Colony settings:
%      n_emp    - # of employed bees                            [100 (default)]
%      n_onl    - # of onlooker bees                            [100 (default)]
%      gen      - bees initialization function                  [200*rand - 100 (default) | @fun(), mex()]
%      phi      - random function in [-1, 1]                    [2*rand - 1 (default) | @fun(), mex()]
%      maxIter  - max non-update sol iter before rejection      [25 (default)]
%      opt_i    - optimal point initialization                  [double(1, dim + dim_g + 1)]
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

%% Internal parameters
tau = 0.5;      % rho updating factor (0 < tau < 1)
gamma = 2;      % rho scaling factor (gamma > 1)

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
if isempty(phi), phi = @() 2*rand - 1; end
if isempty(maxIter), maxIter = 25; end
if isempty(lamLimit), lamLimit = [-1e3, 1e3]; end
if isempty(lagCycle), if ~isempty(g), lagCycle = 10; else, lagCycle = 1; end, end
if isempty(opts)
    opts = struct('nFig', 1, 'showFig', [false, false, false], 'v', true);
end

% Additional parameters
if ~isempty(g)
    dim_g = length(g(zeros(1, dim)));
    L = @(f_f, f_g, l, rho) f_f + 0.5*sum((f_g.*rho + l).^2./rho);
else
    dim_g = 0; g = @(x) 0; 
    L = @(f_f, f_g, l, rho) f_f;
end
lam = zeros(1, dim_g); % Lagrangian multipliers

%% Hive initialization
hive = gen(n_emp + n_onl + 1);
if ~isempty(opt_i) && size(opt_i, 2) == dim + 2*dim_g
    hive(end, :) = opt_i(1, 1:dim);
    lam = opt_i(1, dim+1:dim+dim_g); rho = opt_i(end-dim_g+1:end);
end
n_nup = zeros(size(hive, 1), 1);

% Cost function evalutation
f_f = zeros(size(hive, 1), 1);  f_g = zeros(size(hive, 1), max(1, dim_g));
f_L = zeros(size(hive, 1), 1);
time = [0, 0, 0];
for i = 1:size(hive, 1)
    tic, f_f(i, :) = f(hive(i, 1:dim)); time(1) = time(1) + toc;
    tic, f_g(i, :) = g(hive(i, 1:dim)); time(2) = time(2) + toc;
end

% Optimal solutions
[f_f(end, :), index] = min(f_f);
hive(end, :) = hive(index, :); f_g(end, :) = f_g(index, :);
if isempty(opt_i) || size(opt_i, 2) ~= dim + 2*dim_g
    rho = zeros(size(lam));
    for i = 1:dim_g
        if isempty(opt_i) || size(opt_i, 2) ~= dim + dim_g
            rho(i)= max(1e-6, min(10, 2*abs(f_f(end, :))/((f_g(end, i)).^2)));
        end
    end
end

for i = 1:size(hive, 1)
    tic, f_L(i, :) = L(f_f(i, :), f_g(i, :), lam, rho);
        time(3) = time(3) + toc;
end
f_L(end, :) = f_L(index, :);

%% Plot and verbose
if opts.showFig(1), drawHive(opts.nFig, dim, hive, f, f_f, n_emp, n_onl, ...
        1, lb, ub); end

if opts.v
    fprintf('Cost fun. computation time:  %.2fms\n', 1000*time(1)/size(hive, 1))
    if dim_g
        fprintf('Constraint computation time: %.2fms\n', 1000*time(2)/size(hive, 1))
        fprintf('Lagrangian computation time: %.2fms\n', 1000*time(3)/size(hive, 1))
    end
    fprintf('Estimated tot. comp. time:   %.0fm %.0fs\n',...
        floor((sum(time))*cycle*lagCycle/60),...
        mod((sum(time))*cycle*lagCycle, 60)) % ---------------------------- check estimation
    reply = input('Do you want to continue? Y/N [Y]: ', 's');
    switch reply
        case 'N', ABC_time = 0; opt = [hive(end, :), lam, rho]; return
        otherwise, clear reply time
    end
end

%% Algorithm
lenDisp = 0; tic
for iter1 = 1:lagCycle
    for iter2 = 1:cycle
        if opts.v
            fprintf([repmat('\b', 1, lenDisp)]);
            lenDisp = fprintf('Iter: %d of %d\n', (iter1-1)*cycle + iter2, lagCycle*cycle);
        end

        % Resample onlooker bees
        fit = abs(f_L(1:n_emp, :) - f_L(end, :));
        fit = max(fit) - fit;

        % Probabiloty and desity function
        if sum(fit) < 1e-5, prob = ones(1, length(fit))/length(fit);
        else, prob = fit/sum(fit); end
        dens_prob = cumsum(prob);

        for i = 1:n_onl
            index = find(dens_prob >= rand, 1);
            hive(n_onl + i, :) = hive(index, :);
            f_f(n_onl + i, :) = f_f(index, :); f_g(n_onl + i, :) = f_g(index, :);
            f_L(n_onl + i, :) = f_L(index, :);
        end

        % Move bees
        for i = 1:n_emp + n_onl
            % Choose bee to compare
            k = i; 
            while k == i
                k = randi(n_emp + n_onl, 1);
                j = randi(dim, 1);
            end

            % Optimize solution
            new_sol = hive(i, :);
            new_sol(1, j) = new_sol(1, j) + phi()*(new_sol(1, j) - hive(k, j));

            f_new = f(new_sol); g_new = g(new_sol);
            L_new = L(f_new, g_new, lam, rho);

            % Move if best solution is found
            [hive(i, :), check] = feasPos(new_sol(1, :), hive(i, :), ...
                L_new, f_L(i, :), lb, ub);

            if check
                n_nup(i) = 0;
                f_f(i, :) = f_new; f_g(i, :) = g_new; f_L(i, :) = L_new;
            else
                n_nup(i) = n_nup(i) + 1;
            end
        end

        [f_L(end, :), index] = min(f_L);
        hive(end, :) = hive(index, :);
        f_f(end, :) = f_f(index, :); f_g(end, :) = f_g(index, :);
        
        if opts.showFig(2), drawHive(opts.nFig, dim, hive, f, f_f, n_emp, n_onl, ...
            1, lb, ub); end
        
        % Reject non updated points
        for i = 1:n_emp + n_onl
            if n_nup(i) >= maxIter
                hive(i, :) = gen(1);
                f_f(i, :) = f(hive(i, :)); f_g(i, :) = g(hive(i, :));
                f_L(i, :) = L(f_f(i, :), f_g(i, :), lam, rho);
                n_nup(i, :) = 0;
            end
        end

    end

    % Update Lagrangian
    lam = min(lamLimit(2), max(lamLimit(1), lam + rho.*f_g(index))); % updte lagrangian multipliers
    for i = 1:dim_g
        if max(abs(f_g(index, i))) > tau*max(abs(f_g(end, i))) && max(abs(f_g(index, i))) > 1e-4
            rho(i) = gamma*rho(i);
        elseif max(abs(f_g(index, i))) < 1e-2
            rho(i) = rho(i)/gamma;
        end
    end
    
    if iter1 ~= lagCycle
        % Resample hive
        hive(1:end-1, :) = gen(n_emp + n_onl);
        for i = 1:size(hive, 1)
            f_f(i, :) = f(hive(i, 1:dim)); f_g(i, :) = g(hive(i, 1:dim));
            f_L(i, :) = L(f_f(i, :), f_g(i, :), lam, rho);
        end
        n_nup = zeros(size(hive, 1), 1);
    end
end

ABC_time = toc;
if opts.showFig(3), drawHive(opts.nFig, dim, hive, f, f_f, n_emp, n_onl, ...
        1, lb, ub); end
if dim_g, opt = [hive(end, :), lam, rho];
else, opt = hive(end, :); end

end

%% Functions
function drawHive(nFig, dim, hive, f, f_f, n_emp, n_onl, n_opt, lb, ub)
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
                f_fplot(i, j) = f([x(i, j), y(i, j), hive(1, 3:dim)]);
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