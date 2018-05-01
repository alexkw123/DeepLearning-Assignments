function [W, b, K, rho, m] = InitializeParameters(X_train, y_train, layers)
m = 100;
rho = 0.9;
% initialize parameters
[d, ~] = size(X_train);
K = length(min(y_train):max(y_train));

rng(1);
stanDev = 0.01;
% He initialization
% W1 = sqrt(2/d)*randn(m,d);
% W2 = sqrt(2/m)*randn(K,m);

% init W and b for each layer
W1 = stanDev*randn(m,d);
b1 = zeros(m,1);
W = {W1};
b = {b1};
for i = 2:(layers-1)
    Wi = stanDev*randn(m,m);
    W = [W; {Wi}];
    bi = stanDev*randn(m,1);
    b = [b; {bi}];
end
Wend = stanDev*randn(K,m);
bend = zeros(K,1);
W = [W; {Wend}];
b = [b; {bend}];
end