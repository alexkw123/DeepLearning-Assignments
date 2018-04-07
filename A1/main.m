% read in training, validataion and test data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

% initialize parameters
[d, N] = size(X);
K = length(min(y_train):max(y_train));

mean = 0;
stanDev = 0.01;
W = stanDev*randn(K,d) + mean;
b = stanDev*randn(K,1) + mean;

% test if the ComputeCost and ComputeGradients functions are right
lambda = 0;
P = EvaluateClassifier(X_train(1:100, 1), W(:, 1:100), b);
% function given by professor
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(X_train(1:100, 1), Y_train(:, 1), W(:, 1:100), b, lambda, 1e-6);
% implemented function
[grad_W, grad_b] = ComputeGradients(X_train(1:100, 1), Y_train(:, 1), P, W(:, 1:100), lambda, K);
% relative error
error_b = sqrt((grad_b - ngrad_b).^2)./max(1e-6,sqrt(grad_b.^2)+sqrt(ngrad_b.^2));
error_W = sqrt((grad_W - ngrad_W).^2)./max(1e-6,sqrt(grad_W.^2)+sqrt(ngrad_W.^2));


