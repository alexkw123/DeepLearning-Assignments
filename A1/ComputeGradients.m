function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda, K)

[d, n] = size(X);
grad_W = zeros(K, d);
grad_b = zeros(K, 1);

for i = 1:n
    x = X(:, i);
    y = Y(:, i);
    p = P(:, i);
    g = -(y-p)';
    grad_b = grad_b + g';
    grad_W = grad_W + g' * x';
end

grad_b = grad_b/n;
grad_W = grad_W/n + 2 * lambda * W;

end