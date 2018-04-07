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

% P = EvaluateClassifier(X_train(:, 1:100), W, b);
lambda = 0.8;
Cost = ComputeCost(X_train(:, 1:100), Y_train(:, 1:100), W, b, lambda);
acc = ComputeAccuracy(X_train(:, 1:100), y_train(1:100), W, b);