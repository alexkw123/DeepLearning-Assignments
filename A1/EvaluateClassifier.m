function P = EvaluateClassifier(X, W, b)
% X: dxN
% W: Kxd
% b: Kx1
% apply: (1) s = Wx+b; (2) p = softmax(s)
s = W * X;
s = bsxfun(@plus, s, b);
P = bsxfun(@rdivide, exp(s), sum(exp(s), 1));
end