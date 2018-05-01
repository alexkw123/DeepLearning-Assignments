function [W, b, K, rho] = InitializeParameters(X_train, y_train, layers, hnodes)
% m = 50;
rho = 0.9;
% initialize parameters
[d, ~] = size(X_train);
K = length(min(y_train):max(y_train));

% random init W and b for each layer
% rng(1);
% stanDev = 0.01;
% W = cell(layers);
% b = cell(layers);
% W{1} = stanDev*randn(hnodes(1),d);
% b{1} = zeros(hnodes(1),1);
% for i = 2:(layers-1)
%     W{i} = stanDev*randn(hnodes(i),hnodes(i-1));
%     b{i} = zeros(hnodes(i),1);
% end
% W{layers} = stanDev*randn(K,hnodes(end));
% b{layers} = zeros(K,1);

% He initialization
W = cell(layers);
b = cell(layers);
W{1} = sqrt(2/d)*randn(hnodes(1),d);
b{1} = zeros(hnodes(1),1);
for i = 2:(layers-1)
    W{i} = sqrt(2/hnodes(i-1)) * randn(hnodes(i),hnodes(i-1));
    b{i} = zeros(hnodes(i),1);
end
W{layers} = sqrt(2/hnodes(end))*randn(K,hnodes(end));
b{layers} = zeros(K,1);
end