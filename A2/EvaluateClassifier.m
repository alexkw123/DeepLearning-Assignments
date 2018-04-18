function [P, h, s1] = EvaluateClassifier(X, W, b)
% X: dxN
% W: Kxd
% b: Kx1
% apply: (1) s1=W1x+b1 (2) h=max(0,s1) (3) s=W2h+b2 (4) p = softmax(s)
s1 = W{1} * X;
s1 = bsxfun(@plus, s1, b{1});
% ReLU
% h = bsxfun(@max, 0, s1);
% tanh
h = arrayfun(@(x) tanh(x),s1);
s = W{2} * h;
s = bsxfun(@plus, s, b{2});
P = bsxfun(@rdivide, exp(s), sum(exp(s), 1));
end