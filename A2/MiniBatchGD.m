function [Wstar, bstar] = MiniBatchGD(X, Y, eta, W, b, lambda, m)

[P, h, s1] = EvaluateClassifier(X, W, b);
[K, ~] = size(Y);
[grad_W, grad_b] = ComputeGradients(X, Y, P, h, s1, W, lambda, K, m);

W1star = W{1} - eta * grad_W{1};
W2star = W{2} - eta * grad_W{2};
Wstar = {W1star, W2star};

b1star = b{1} - eta * grad_b{1};
b2star = b{2} - eta * grad_b{2};
bstar = {b1star, b2star};

end