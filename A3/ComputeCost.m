function J = ComputeCost(X, Y, W, b, lambda, varargin)
% equation 5:
% X: an image (dxn)
% Y: one-hot label for the column (Kxn)
[~, n] = size(X);
% [P, ~, ~, ~, ~, ~] = EvaluateClassifier(X, W, b, varargin);
flag = isempty(varargin);
if flag
    [P, ~, ~, ~, ~, ~] = EvaluateClassifier(X, W, b);
else
    [P, ~, ~, ~, ~, ~] = EvaluateClassifier(X, W, b, varargin);
end
temp = 0;
% disp(length(W));
for i = 1:length(W)
    temp = temp + sum(sum((W{i}.^2),1));
end
J = -sum(log(sum(Y.*P, 1)))/n + lambda*temp;
end