function [W, b, K, rho, m] = InitializeParameters(X_train, y_train)
m = 50;
rho = 0.9;
% initialize parameters
[d, ~] = size(X_train);
K = length(min(y_train):max(y_train));

% rng(1);
stanDev = 0.001;
W1 = stanDev*randn(m,d);
W2 = stanDev*randn(K,m);
b1 = zeros(m,1);
b2 = zeros(K,1);

W = {W1, W2};
b = {b1, b2};
end