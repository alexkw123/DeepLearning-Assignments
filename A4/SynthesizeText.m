function [Y] = SynthesizeText(RNN, h0, X)
% h0: the hidden state at time 0
% x0: the first (dummy) input vector to RNN (it can be some character like afull-stop)
% n: the length of the sequence to generate

% P = zeros(RNN.K, n);
% H = zeros(RNN.m, n);
[K,n] = size(X);
% [m,~] = size(h0);
Y = zeros(K, n);
% P = zeros(K, n);
% H = zeros(m, n);
h = h0;
x = zeros(K, 1);
x(12) = 1;
for t = 1:n
    a = RNN.W * h + RNN.U * x + RNN.b;
    h = tanh(a);
    o = RNN.V * h + RNN.c;
    p = bsxfun(@rdivide, exp(o), sum(exp(o), 1));
    
    cp = cumsum(p);  % compute the vector containing the cumulative sum of the probabilities
    a = rand;
    ixs = find(cp-a >0);
    ii = ixs(1);
    
    Y(ii, t) = 1;
%     P(:, t) = p;
%     H(:, t) = h;
end