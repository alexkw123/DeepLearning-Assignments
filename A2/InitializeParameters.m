function [W, b, K, rho, m] = InitializeParameters(X_train, y_train)
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
W1 = stanDev*randn(m,d);
W2 = stanDev*randn(K,m);
b1 = zeros(m,1);
b2 = zeros(K,1);

W = {W1, W2};
b = {b1, b2};
end