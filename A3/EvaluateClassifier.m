function [P, s, h] = EvaluateClassifier(X, W, b)
% X: dxN
% W: Kxd
% b: Kx1
% apply: (1) s1=W1x+b1 (2) h=max(0,s1) (3) s=W2h+b2 (4) p = softmax(s)
s = {};
h = {};
x = X;
k = length(W);
for i = 1:(k-1)
    si = W{i} * x;
    si = bsxfun(@plus, si, b{i});
    s = [s; {si}];
    % ReLU
    x = bsxfun(@max, 0, si);
    h = [h; {x}];
    % tanh
    % h = arrayfun(@(x) tanh(x),s1);
end
send = W{k} * x;
send = bsxfun(@plus, send, b{k});
s = [s; {send}];
P = bsxfun(@rdivide, exp(send), sum(exp(send), 1));
end