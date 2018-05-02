function [P, s, h] = EvaluateClassifier(X, W, b)
% X: dxN
% W: Kxd
% b: Kx1
% apply: (1) s1=W1x+b1 (2) h=max(0,s1) (3) s=W2h+b2 (4) p = softmax(s)
x = X;
[~, n] = size(x);
k = length(W);
s = cell(k);
miu = cell(k);
vari = cell(k);
h = cell(k-1);
for i = 1:(k-1)
    s{i} = bsxfun(@plus, W{i} * x, b{i});
    % batch normalize
    miu{i} = sum(s{i}) / n;
    temp = var(s{i}, 0, 2);
    vari{i} = temp * (n-1) / n;
    si = BatchNormalize(s{i}, miu{i}, vari{i});
    % ReLU
    x = bsxfun(@max, 0, si);
    h{i} = x;
    % tanh
    % h = arrayfun(@(x) tanh(x),s{i});
end
s{k} = bsxfun(@plus, W{k} * x, b{k});

P = bsxfun(@rdivide, exp(s{k}), sum(exp(s{k}), 1));
end


function [s] = BatchNormalize(s, mu, v)
s = cellfun(@(x) (v.^(-0.5))'.*(x-mu), s, 'UniformOutput', false);
end