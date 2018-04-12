function J = ComputeCost(X, Y, W, b, lambda)
% equation 5:
% X: an image (dxn)
% Y: one-hot label for the column (Kxn)
[~, n] = size(X);
s = EvaluateClassifier(X, W, b);
f = zeros(n,1);
% calculate max(0,s-sy+1)
for i = 1:n
    [~, yi] = max(Y(:, i));
    temp = bsxfun(@minus, s(:,i), s(yi,i));
    temp = bsxfun(@plus, temp, 1);
    l = bsxfun(@max, 0, temp);
    f(i,1) = sum(l);
end
J = sum(f)/n + lambda*sum(sum((W.^2),1));
end