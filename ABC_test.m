%% Initialization

dim     = 2;
f       = @(x) -418.9829*size(x, 2) + sum(x.*sin(sqrt(abs(x))));
lb      = -10*ones(1, dim);
ub      = 10*ones(1, dim);
g       = []; % @(x) [x(1).^2 - 1, x(2).^2 - 1];
type    = 'min';
cycle   = 100;
n_emp   = 100;
n_onl   = 100;
gen     = @(n, dim) 20*rand(n, dim) - 10;
phi     = @() 2*rand - 1;
maxIter = [];
n_opt   = [];
tol     = [];
hive_i  = [];
hive    = [];

%% Run function
tic
[opt, hive] = ABC(dim, f, lb, ub, g, type, cycle, ...
                 n_emp, n_onl, gen, phi, maxIter, ...
                 n_opt, tol, hive);
ABC_time = toc