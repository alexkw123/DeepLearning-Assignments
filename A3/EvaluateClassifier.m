function [P, s, sp, h, mu, v] = EvaluateClassifier(X, W, b, varargin)
% X: dxN
% W: Kxd
% b: Kx1
% apply: (1) s1=W1x+b1 (2) h=max(0,s1) (3) s=W2h+b2 (4) p = softmax(s)
x = X;
[~, n] = size(x);
k = length(W);  % number of layers
h = cell(k-1,1);
s = cell(k,1);
sp = cell(k-1,1);

flag = isempty(varargin); % moving average
if flag
  mu = cell(k-1, 1);
  v = cell(k-1, 1);
else
  mu = varargin{1}.mu;
  v = varargin{1}.v;
end

for i = 1:(k-1)
    s{i} = bsxfun(@plus, W{i} * x, b{i});
    % batch normalize
    if flag
        mu{i} = mean(s{i}, 2);
        v{i} = var(s{i}, 0, 2)*(n-1)/n;
    end
    sp{i} = diag(v{i}.^(-0.5))*bsxfun(@minus, s{i}, mu{i});
    % ReLU
%     x = bsxfun(@max, 0, s{i});
    x = bsxfun(@max, 0, sp{i});
    h{i} = x;
    % tanh
    % h = arrayfun(@(x) tanh(x),s{i});
end
s{k} = bsxfun(@plus, W{k} * x, b{k});

P = bsxfun(@rdivide, exp(s{k}), sum(exp(s{k}), 1));
end