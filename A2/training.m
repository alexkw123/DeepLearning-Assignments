% ------- training process ---------
% read in training, validataion and test data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

% [X_train1, Y_train1, y_train1] = LoadBatch('data_batch_1.mat');
% [X_train2, Y_train2, y_train2] = LoadBatch('data_batch_2.mat');
% [X_train3, Y_train3, y_train3] = LoadBatch('data_batch_3.mat');
% [X_train4, Y_train4, y_train4] = LoadBatch('data_batch_4.mat');
% [X_train5, Y_train5, y_train5] = LoadBatch('data_batch_5.mat');
% X_train = [X_train1 X_train2 X_train3 X_train4 X_train5];
% Y_train = [Y_train1 Y_train2 Y_train3 Y_train4 Y_train5];
% y_train = [y_train1 y_train2 y_train3 y_train4 y_train5];
% 
% [~, N] = size(X_train);
% 
% split = N - 999;
% X_val = X_train(:, split:N);
% Y_val = Y_train(:, split:N);
% y_val = y_train(split:N);
% 
% X_train(:, split:N) = [];
% Y_train(:, split:N) = [];
% y_train(split:N) = [];

% X_train = X_train(:,1:1000);
% Y_train = Y_train(:,1:1000);
% y_train = y_train(1:1000);

% transform training data to have zero mean
% mean_X = mean(X_train, 2);
% X_train = X_train - repmat(mean_X, [1, size(X_train, 2)]);
% % subtract it from the input vectors in the validation and test sets
% X_val = X_val - repmat(mean_X, [1, size(X_val, 2)]);
% X_test = X_test - repmat(mean_X, [1, size(X_test, 2)]);

% initialize the network
% rng(3);
[W, b, K, rho, m] = InitializeParameters(X_train, y_train);

% set training parameters
% n_epochs=10; n_batch=250;  % 0.4442   %0.5005

% lambda = 3.24e-05; eta = 0.108993;  % 0.4576
% lambda = 9.21E-05; eta = 0.135561;  % 0.4351

% n_epochs = 10; n_batch = 100; lambda = 2.24e-05; eta = 0.108993;
% n_epochs = 10; n_batch = 100; lambda = 1.24e-05; eta = 0.118811;
% n_epochs = 10; n_batch = 100; lambda = 3.91e-05; eta = 0.119100;
% n_epochs = 10; n_batch = 100; lambda = 3.91e-05; eta = 0.119100;

%  for lab 3 final part
% n_epochs = 10; n_batch = 100; lambda = 0; eta = 0.002; % 2layer_1  0.2715
% n_epochs = 10; n_batch = 100; lambda = 0; eta = 0.02;  % 2layer_2  0.4072
n_epochs = 10; n_batch = 100; lambda = 0; eta = 0.4;   % 2layer_3  0.1718

% training
[W, b, cost_train, cost_val] = MiniBatchGD(X_train, Y_train, X_val, Y_val, W, b, lambda, n_epochs, n_batch, eta, m, rho);

% calculate the accuracy
% [X, ~, y] = LoadBatch('test_batch.mat'); % 1

acc = ComputeAccuracy(X_test, y_test, W, b);
disp(acc);

% acc = ComputeAccuracy(X_train, y_train, W, b);
% disp(acc);

% acc_best = ComputeAccuracy(X_test, y_test, best_W, best_b);
% disp(acc_best);

% plot the cost function
inds = 1:n_epochs;
plot(inds, cost_train, inds, cost_val);
% disp(cost_train);