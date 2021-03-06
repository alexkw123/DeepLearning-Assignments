% ------- training.m ---------
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
% 
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
layers = 2;
hnodes = [50];
rng(1);
[W, b, K, rho] = InitializeParameters(X_train, y_train, layers, hnodes);

% set training parameters
% n_epochs = 10; n_batch = 100; lambda = 3.91e-05; eta = 0.119100;
% n_epochs = 10; n_batch = 100; lambda = 0; eta = 0.002; % 3layer_1  0.1000  0.1823 | 0.4277
% n_epochs = 10; n_batch = 100; lambda = 0; eta = 0.02;   % 3layer_2  0.1948  (0.3228  0.2784 0.3103) | 0.4182
% n_epochs = 10; n_batch = 100; lambda = 0; eta = 0.4;   % 3layer_3  0.2478  (0.2977 0.3007) | 0.4079

% coarse search best three
% n_epochs = 20; n_batch = 100;         % 10 epoch:random init, he init, 20 epoch
% lambda = 2.82E-06; eta = 0.00591887;  %0.4237, 0.4099, 0.4155
% lambda = 1.48E-05; eta = 0.00902224;  %0.4244, 0.4024,
% lambda = 2.50E-06; eta = 0.0180744;   %0.4184, , 0.4167

% fine search best three 2
% lambda = 5.91E-05; eta = 0.00266124;    %0.4318
% lambda = 1.00E-05; eta = 0.00493538;     %0.4310
% lambda = 0.000897411; eta = 0.00179585;    %0.4210

% 2 layer network with batch and without batch
% n_epochs = 10; n_batch = 100; lambda = 0; eta = 0.002; % 2layer_1  0.3871
% n_epochs = 10; n_batch = 100; lambda = 0; eta = 0.02;  % 2layer_2  0.4051
% n_epochs = 10; n_batch = 100; lambda = 0; eta = 0.4;   % 2layer_3  0.3880

% training
[W, b, cost_train, cost_val, ma] = MiniBatchGD(X_train, Y_train, X_val, Y_val, W, b, lambda, n_epochs, n_batch, eta, rho);
% disp(cost_val);

% calculate the accuracy

acc = ComputeAccuracy(X_test, y_test, W, b, ma);
disp(acc);

% acc = ComputeAccuracy(X_train, y_train, W, b);
% disp(acc);

% acc_best = ComputeAccuracy(X_test, y_test, best_W, best_b);
% disp(acc_best);

% plot the cost function
inds = 1:n_epochs;
plot(inds, cost_train, inds, cost_val);