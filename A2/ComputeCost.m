function J = ComputeCost(X, Y, W, b, lambda)
% equation 5:
% X: an image (dxn)
% Y: one-hot label for the column (Kxn)
[~, n] = size(X);
P = EvaluateClassifier(X, W, b);
W1 = W{1}; W2 = W{2};
J = -sum(log(sum(Y.*P, 1)))/n + lambda*(sum(sum((W1.^2),1))+sum(sum((W2.^2),1)));
end