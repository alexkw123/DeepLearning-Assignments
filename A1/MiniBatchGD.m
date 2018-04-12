function [Wstar, bstar] = MiniBatchGD(X, Y, eta, W, b, lambda)

% P = EvaluateClassifier(X, W, b);
% [K, ~] = size(Y);
s = EvaluateClassifier(X, W, b);
[grad_W, grad_b] = ComputeGradients(X, Y, W, b, lambda, s);

Wstar = W - eta * grad_W;
bstar = b - eta * grad_b;

end