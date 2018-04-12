function [grad_W, grad_b] = ComputeGradients(X, Y, W, b, lambda, s)

[~, n] = size(X);
grad_W = W;
grad_b = b;

for i = 1:n
    x = X(:, i);
    [~, yi] = max(Y(:, i));
    
    temp = bsxfun(@minus, s(:,i), s(yi,i));
    temp = bsxfun(@plus, temp, 1);
    l = bsxfun(@max, 0, temp);
    l = double(l>0);
    l(yi) = -(sum(l)-1);
    
    g = l';
    grad_b = grad_b + g';
    grad_W = grad_W + g' * x';
end

grad_b = grad_b/n;
grad_W = grad_W/n + 2 * lambda * W;

end