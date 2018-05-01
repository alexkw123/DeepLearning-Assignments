function [grad_W, grad_b] = ComputeGradients(X, Y, P, S, H, W, lambda, K, m)

[d, n] = size(X);
k = length(W);

grad_W = {zeros(m, d)};
grad_b = {zeros(m, 1)};
for i = 2:(k-1)
    grad_W = [grad_W, {zeros(m, m)}];
    grad_b = [grad_b, {zeros(m, 1)}];
end
grad_W = [grad_W, {zeros(K, m)}];
grad_b = [grad_b, {zeros(K, 1)}];

for i = 1:n
    x = X(:, i);
    y = Y(:, i);
    p = P(:, i);
    
    
    g = -(y-p)';
    
    for j = k:-1:2
        grad_b{j} = grad_b{j} + g';
        h = H{j-1}(:, i);
        
        disp(j);
        disp(size(W{j}));
        disp(size(g' * h'));
        
        grad_W{j} = grad_W{j} + g' * h';
        g = g * W{j};
        
        s = S{j-1}(:, i);
        v = double(s>0);
        g = g * diag(v);
    end
    
    grad_b{1} = grad_b{1} + g';
    grad_W{1} = grad_W{1} + g' * x';
end

for i = 1:k
    grad_W{i} = grad_W{i}/n + 2 * lambda * W{i};
    grad_b{i} = grad_b{i}/n;
end

end