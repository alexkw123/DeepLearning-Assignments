function [Wstar, bstar] = MiniBatchGD(X, Y, eta, W, b, lambda)

P = EvaluateClassifier(X, W, b);
[K, ~] = size(Y);
[grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda, K);

Wstar = W - eta * grad_W;
bstar = b - eta * grad_b;

end