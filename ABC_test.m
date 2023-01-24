clc
%% Initialization

dim     = 100;
f       = @(x) 418.9829*size(x, 2) + sum(x.*sin(sqrt(abs(x))));
lb      = -10*ones(1, dim);
ub      = 10*ones(1, dim);
g       =  []; % @(x) [x(1) - 1, x(2) - 1];
type    = 'max';
cycle   = 100;
n_emp   = 10000;
n_onl   = 10000;
gen     = @(n, dim) 2*max([ub, abs(lb)])*rand(n, dim) - max([ub, abs(lb)]);
phi     = @() 2*rand - 1;
maxIter = inf;
nEqLag  = 10;
n_opt   = [];
tol     = 100;
hive    = [];

%% Run function
[opt, hive, ABC_time] = ABC(dim, f, lb, ub, g, type, ...
                n_emp, n_onl, cycle, gen, phi, maxIter, ...
                nEqLag, n_opt, tol, hive);

fprintf('----------------------------------------\n')
fprintf('Total computation time:      %.0fm %.0fs\n\n',...
    floor(ABC_time/60), mod(ABC_time, 60))