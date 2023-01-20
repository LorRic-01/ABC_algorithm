%% Initialization

dim     = 2;
f       = @(x) sum(x(:, 1:2).^4, 2); % 418.9829*size(x, 2) + sum(x.*sin(sqrt(abs(x))))
lb      = -10*ones(1, dim);
ub      = 10*ones(1, dim);
g       =  @(x) [x(:, 1) - 1, x(:, 2) - 1];
type    = 'min';
cycle   = 100;
n_emp   = 100;
n_onl   = 100;
gen     = @(n, dim) 2*max([ub, abs(lb)])*rand(n, dim) - max([ub, abs(lb)]);
phi     = @() 2*rand - 1;
maxIter = inf;
n_opt   = [];
tol     = 0.5;
hive    = [];

%% Run function
[opt, hive, ABC_time] = ABC(dim, f, lb, ub, g, type, cycle, ...
                n_emp, n_onl, gen, phi, maxIter, ...
                n_opt, tol, hive);
ABC_time