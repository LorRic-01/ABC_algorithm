%% Initialization
dim      = 5;
f        = @(x) sum(x.^2);
lb       = -10*ones(1, dim);
ub       = 10*ones(1, dim);
n_emp    = 100;
n_onl    = 100;
cycle    = 200;
gen      = @(n) (ub-lb).*rand(n, dim) + lb;
phi      = @() 2*rand - 1;
maxIter  = inf;
g        = @(x) x - 1;
lamLimit = [];
lagCycle = 10;
opts     = struct('nFig', 1, 'showFig', [true, false, true], 'v', true);
opt     = [];

%% Run optimization
[opt, hive, ABC_time] = ABC(dim, f, lb, ub, ...
                n_emp, n_onl, cycle, gen, phi, maxIter,...
                g, lamLimit, lagCycle,...
                opts, opt);

fprintf('----------------------------------------\n')
fprintf('Total computation time:      %.0fm %.0fs\n\n',...
    floor(ABC_time/60), mod(ABC_time, 60))