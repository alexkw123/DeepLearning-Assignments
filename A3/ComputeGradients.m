function [grad_W, grad_b] = ComputeGradients(X, Y, P, S, H, W, lambda, b, Sp, mu, v)

[~, n] = size(X);
k = length(W);

grad_W = cell(k,1);
grad_b = cell(k,1);
for i = 1:k
    grad_W{i} = zeros(size(W{i}));
    grad_b{i} = zeros(size(b{i}));
end

g = cell(n,1);
for i = 1:n
    y = Y(:, i);
    p = P(:, i);
    
    g{i} = -(y-p)';
end
    
% the last layer
for i = 1:n
    grad_b{k} = grad_b{k} + g{i}';
    grad_W{k} = grad_W{k} + g{i}' * H{k}(:, i)';
end
grad_b{k} = grad_b{k}/n;
grad_W{k} = grad_W{k}/n + 2 * lambda * W{k};

% Propagate the gradient vector g to the previous layer
for i = 1:n
    g{i} = g{i} * W{k};
    g{i} = g{i} * diag(double(Sp{k-1}(:, i)>0));
end

% the other layers
for j = (k-1):-1:1
    g = BatchNormBackPass(g, S{j}, mu{j}, v{j});
    for i = 1:n
        grad_b{j} = grad_b{j} + g{i}';
        grad_W{j} = grad_W{j} + g{i}' * H{j}(:, i)';
    end
    grad_b{j} = grad_b{j}/n;
    grad_W{j} = grad_W{j}/n + 2 * lambda * W{j};
    
    if j > 1
        for i = 1:n
            g{i} = g{i} * W{j};
            g{i} = g{i} * diag(double(Sp{j-1}(:, i)>0));
        end
    end
end

end