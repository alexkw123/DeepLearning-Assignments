function J = ComputeCost(X, Y, W, b, lambda)
% equation 5:
% X: an image (dxn)
% Y: one-hot label for the column (Kxn)
[~, n] = size(X);
P = EvaluateClassifier(X, W, b);
J = -log(sum(Y'*P, 1))/n + lambda*sumsqr(W);
end

function [s,n] = sumsqr(x)
x = x(:);
t = isfinite(x);
s = sum(x(t).^2);
n = sum(t);
end