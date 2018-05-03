function [P, s, sp, h, mu, v] = EvaluateClassifier(X, W, b)
% X: dxN
% W: Kxd
% b: Kx1
% apply: (1) s1=W1x+b1 (2) h=max(0,s1) (3) s=W2h+b2 (4) p = softmax(s)
x = X;
[~, n] = size(x);
k = length(W);  % number of layers
h = cell(k,1);
s = cell(k,1);
sp = cell(k,1);

mu = cell(k, 1);
v = cell(k, 1);

for i = 1:(k-1)
    s{i} = bsxfun(@plus, W{i} * x, b{i});
    % batch normalize
    mu{i} = mean(s{i}, 2);
    v{i} = var(s{i}, 0, 2) * (n-1) / n;
    sp{i} = diag(v{i}.^(-0.5))*(s{i}-mu{i});
    % ReLU
    h{i} = x;
    x = bsxfun(@max, 0, sp{i});
    % tanh
    % h = arrayfun(@(x) tanh(x),s{i});
end
s{k} = bsxfun(@plus, W{k} * x, b{k});
h{k} = x;

P = bsxfun(@rdivide, exp(s{k}), sum(exp(s{k}), 1));
end