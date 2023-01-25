clc
%% Initialization

dim     = 2;
f       = @(x) -418.9829*size(x, 2) - sum(x.*sin(sqrt(abs(x))));
lb      = -500*ones(1, dim);
ub      = 500*ones(1, dim);
g       = []; % @(x) [x(1) - 1];
cycle   = 1000;
n_emp   = 100;
n_onl   = 100;
gen     = @(n, dim) 2*max([ub, abs(lb)])*rand(n, dim) - max([ub, abs(lb)]);
phi     = @() 2*rand - 1;
maxIter = inf;
nEqLag  = 10;
n_opt   = [];
tol     = 100;
option  = struct('nFig', 1, 'showFig', [false, false, true], 'v', true);
hive    = [];

%% Run function
[opt, hive, ABC_time] = ABC(dim, f, lb, ub, g, ...
                n_emp, n_onl, cycle, gen, phi, maxIter, ...
                nEqLag, n_opt, tol, option, hive);

fprintf('----------------------------------------\n')
fprintf('Total computation time:      %.0fm %.0fs\n\n',...
    floor(ABC_time/60), mod(ABC_time, 60))