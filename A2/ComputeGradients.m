function [grad_W, grad_b] = ComputeGradients(X, Y, P, h, s1, W, lambda, K, m)

[d, n] = size(X);
W1 = W{1}; W2 = W{2};
grad_W1 = zeros(m, d); grad_W2 = zeros(K, m);
grad_b1 = zeros(m, 1); grad_b2 = zeros(K, 1);

for i = 1:n
    x = X(:, i);
    y = Y(:, i);
    p = P(:, i);
    
    g = -(y-p)';
    
%     disp(size(g'));
%     disp(size(h'));
%     disp(size(grad_W2));
    grad_b2 = grad_b2 + g';
    grad_W2 = grad_W2 + g' * h';
    
    g = g * W2;
    [~, M] = size(g);
    diag = zeros(M);
    for j = 1:M
        diag(j,j) = double(s1(j)>0);
    end
    g = g * diag;
    
    grad_b1 = grad_b1 + g';
    grad_W1 = grad_W1 + g' * x';
end

grad_b1 = grad_b1/n;
grad_b2 = grad_b2/n;
grad_W1 = grad_W1/n + 2 * lambda * W1;
grad_W2 = grad_W2/n + 2 * lambda * W2;

grad_W = {grad_W1, grad_W2};
grad_b = {grad_b1, grad_b2};

end