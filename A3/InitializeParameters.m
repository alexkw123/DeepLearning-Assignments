function [W, b, K, rho] = InitializeParameters(X_train, y_train, layers, hnodes)
% m = 50;
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
W1 = stanDev*randn(hnodes(1),d);
b1 = zeros(hnodes(1),1);
W = {W1};
b = {b1};
for i = 2:(layers-1)
    Wi = stanDev*randn(hnodes(i),hnodes(i-1));
    W = [W; {Wi}];
    bi = stanDev*randn(hnodes(i),1);
    b = [b; {bi}];
end
Wend = stanDev*randn(K,hnodes(end));
bend = zeros(K,1);
W = [W; {Wend}];
b = [b; {bend}];
end