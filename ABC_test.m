% Benchmark functions
% @(x) -20*exp(-0.2*sqrt(0.5*(x(1)^2 + x(2)^2))) - exp(0.5*(cos(2*pi*x(1)) + cos(2*pi*x(2)))) + exp(1) + 20;
% @(x) sum(x.^4 - 16*x.^2 + 5*x)/2;

%% Initialization
dim      = 2;
f        = @(x) sum(x.^2);
lb       = -5*ones(1, dim);
ub       = 5*ones(1, dim);
g        = @(x) x(1) - 1;
n_emp    = 100;
n_onl    = 100;
cycle    = 100;
gen      = @(n) (ub-lb).*rand(n, dim) + lb;
phi      = @(n) 2*rand(1, n) - 1;
maxIter  = 25;
n_opt    = [];
tol      = 2;
opts     = struct('nFig', 1, 'showFig', [true, false, true], 'v', true);
hive     = [];

%% Run optimization
[opt, hive, ABC_time] = ABC(dim, f, lb, ub, g, ...
                n_emp, n_onl, cycle, gen, phi, maxIter,...
                n_opt, tol, opts, hive);

fprintf('----------------------------------------\n')
fprintf('Total computation time:      %.0fm %.0fs\n\n',...
    floor(ABC_time/60), mod(ABC_time, 60))