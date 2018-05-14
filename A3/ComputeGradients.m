% ------- ComputeGradients.m ---------

function [grad_W, grad_b, ma] = ComputeGradients(X, Y, W, lambda, b, varargin)

[P, S, Sp, H, mu, v] = EvaluateClassifier(X, W, b);

alpha = 0.99;
% exponential moving average
if isempty(varargin)
    ma.mu = mu;
    ma.v = v;
else
    ma.mu = varargin{1}.mu;
    ma.v = varargin{1}.v;
    for e = 1:length(mu)
        ma.mu{e} = alpha*ma.mu{e} + (1-alpha)*mu{e};
        ma.v{e} = alpha*ma.v{e} + (1-alpha)*v{e};
    end
end

[~, n] = size(X);
k = length(W);

grad_W = cell(k,1);
grad_b = cell(k,1);
for i = 1:k
    grad_W{i} = zeros(size(W{i}));
    grad_b{i} = zeros(size(b{i}));
end

g = cell(n,1);
for j = k:(-1):1
    if j == k
        for i = 1:n
            g{i} = -(Y(:, i)-P(:, i))';
        end
    end
    
    if j == 1
        x = X;
    else
        x = H{j-1};
    end
    
    for i = 1:n
        grad_b{j} = grad_b{j} + g{i}';
        grad_W{j} = grad_W{j} + g{i}' * x(:, i)';
        
        if j ~= 1
            g{i} = g{i} * W{j};
            g{i} = g{i} * diag(double(Sp{j-1}(:, i)>0));
        end
    end
    if j ~= 1
        g = BatchNormBackPass(g, S{j-1}, mu{j-1}, v{j-1});
    end
    grad_b{j} = grad_b{j}/n;
    grad_W{j} = grad_W{j}/n + 2 * lambda * W{j};
end

end