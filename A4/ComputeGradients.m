% ------- ComputeGradients.m ---------

function [grads] = ComputeGradients(X, Y, RNN, P, H, h0)

for f = fieldnames(RNN)'
    grads.(f{1}) = zeros(size(RNN.(f{1})));
end

[K,t] = size(X);
[m,~] = size(H);
do = zeros(t, K);
da = zeros(t, m);
dh = zeros(t, m);

for i = 1:t
    y = Y(:, i);
    p = P(:, i);
    h = H(:, i);
    
    g = -(y-p)';
    do(i, :) = g;
    grads.V = grads.V + g' * h';
    grads.c = grads.c + g';
end

dh(t, :) = do(t, :) * RNN.V;
da(t, :) = dh(t, :) * diag(1 - (H(:, t)).^2);
for i = (t-1):(-1):1
    dh(i, :) = do(i, :) * RNN.V + da(i+1, :) * RNN.W;
    da(i, :) = dh(i, :) * diag(1 - (H(:, i)).^2);
end

for i = 1:t
    if i == 1
        grads.W = grads.W + da(i, :)' * h0';
    else
        grads.W = grads.W + da(i, :)' * H(:, i-1)';
    end
    grads.U = grads.U + da(i, :)' * X(:, i)';
    grads.b = grads.b + da(i, :)';
end
end