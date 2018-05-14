function [grad_W, grad_b] = ComputeGradients(X, Y, W, lambda, b)

[P, S, Sp, H, mu, v] = EvaluateClassifier(X, W, b);

[~, n] = size(X);
k = length(W);
eps = 1e-4;

grad_W = cell(k,1);
grad_b = cell(k,1);
for i = 1:k
    grad_W{i} = zeros(size(W{i}));
    grad_b{i} = zeros(size(b{i}));
end

% for i = 1:n
%     x = X(:, i);
%     y = Y(:, i);
%     p = P(:, i);
%     
%     
%     g = -(y-p)';
%     
%     for j = k:-1:2
%         grad_b{j} = grad_b{j} + g';
%         h = H{j-1}(:, i);
%         
%         disp(size(grad_W{j}));
%         disp(size(g'));
%         disp(size(h'));
%         grad_W{j} = grad_W{j} + g' * h';
%         g = g * W{j};
%         
%         s = S{j-1}(:, i);
%         v = double(s>0);
%         g = g * diag(v);
%     end
%     
%     grad_b{1} = grad_b{1} + g';
%     grad_W{1} = grad_W{1} + g' * x';
% end
% 
% for i = 1:k
%     grad_W{i} = grad_W{i}/n + 2 * lambda * W{i};
%     grad_b{i} = grad_b{i}/n;
% end

% batch, the last layer
% g = cell(n,1);
% for i = 1:n
%     y = Y(:, i);
%     p = P(:, i);
%     
%     g{i} = -(y-p)';
%     grad_b{k} = grad_b{k} + g{i}';
%     grad_W{k} = grad_W{k} + g{i}' * H{k}(:, i)';
% end
% 
% grad_b{k} = grad_b{k}/n;
% grad_W{k} = grad_W{k}/n + 2 * lambda * W{k};

% Propagate the gradient vector g to the previous layer

g = cell(n,1);
% the other layers
for j = k:(-1):1
    if j == k
        for i = 1:n
            g{i} = -(Y(:, i)-P(:, i))';
        end
    end
    
    if j == 1
        x = X;
    else
        x = H{j-1};
    end
    
    for i = 1:n
        grad_b{j} = grad_b{j} + g{i}';
        grad_W{j} = grad_W{j} + g{i}' * x(:, i)';
        
        if j ~= 1
            g{i} = g{i} * W{j};
            g{i} = g{i} * diag(double(Sp{j-1}(:, i)>0));
        end
    end
    if j ~= 1
        g = BatchNormBackPass(g, S{j-1}, mu{j-1}, v{j-1}, eps);
    end
    grad_b{j} = grad_b{j}/n;
    grad_W{j} = grad_W{j}/n + 2 * lambda * W{j};
end

end