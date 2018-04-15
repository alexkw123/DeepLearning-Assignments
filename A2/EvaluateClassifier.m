function [P, h, s1] = EvaluateClassifier(X, W, b)
% X: dxN
% W: Kxd
% b: Kx1
% apply: (1) s1=W1x+b1 (2) h=max(0,s1) (3) s=W2h+b2 (4) p = softmax(s)
W1 = W{1};
W2 = W{2};
b1 = b{1};
b2 = b{2};

s1 = W1 * X;
s1 = bsxfun(@plus, s1, b1);
h = bsxfun(@max, 0, s1);
s = W2 * h;
s = bsxfun(@plus, s, b2);
P = bsxfun(@rdivide, exp(s), sum(exp(s), 1));
end