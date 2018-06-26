% ------- ForwardPass.m ---------

function [P, H, loss] = ForwardPass(RNN, h0, X, Y)
% h0: the hidden state at time 0
% x0: the first (dummy) input vector to RNN (it can be some character like afull-stop)
% n: the length of the sequence to generate

% P = zeros(RNN.K, n);
% H = zeros(RNN.m, n);
[K,n] = size(X);
[m,~] = size(h0);
P = zeros(K, n);
H = zeros(m, n);
h = h0;
for t = 1:n
    x = X(:, t);
    a = RNN.W * h + RNN.U * x + RNN.b;
    h = tanh(a);
    o = RNN.V * h + RNN.c;
    p = bsxfun(@rdivide, exp(o), sum(exp(o)));
    
    P(:, t) = p;
    H(:, t) = h;
end

loss = -sum(log(sum(Y.*P, 1)));
end