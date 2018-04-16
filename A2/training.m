% ------- training process ---------
% read in training, validataion and test data
% [X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
% [X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

[X_train1, Y_train1, y_train1] = LoadBatch('data_batch_1.mat');
[X_train2, Y_train2, y_train2] = LoadBatch('data_batch_2.mat');
[X_train3, Y_train3, y_train3] = LoadBatch('data_batch_3.mat');
[X_train4, Y_train4, y_train4] = LoadBatch('data_batch_4.mat');
[X_train5, Y_train5, y_train5] = LoadBatch('data_batch_5.mat');
X_train = [X_train1 X_train2 X_train3 X_train4 X_train5];
Y_train = [Y_train1 Y_train2 Y_train3 Y_train4 Y_train5];
y_train = [y_train1 y_train2 y_train3 y_train4 y_train5];

[~, N] = size(X_train);

split = N - 999;
X_val = X_train(:, split:N);
Y_val = Y_train(:, split:N);
y_val = y_train(split:N);

X_train(:, split:N) = [];
Y_train(:, split:N) = [];
y_train(split:N) = [];

% transform training data to have zero mean
mean_X = mean(X_train, 2);
X_train = X_train - repmat(mean_X, [1, size(X_train, 2)]);
% subtract it from the input vectors in the validation and test sets
X_val = X_val - repmat(mean_X, [1, size(X_val, 2)]);
X_test = X_test - repmat(mean_X, [1, size(X_test, 2)]);

% initialize the network
% rng(3);
[W, b, K, rho, m] = InitializeParameters(X_train, y_train);

% set training parameters
n_epochs=30; n_batch=230;  % 0.4442   %0.5005
% n_epochs=30; n_batch=300;

lambda = 2.24e-05; eta = 0.108993;
% lambda = 1e-5; eta = 0.025;
% lambda = 3.81e-05; eta = 0.108799;  %0.4008
% lambda = 4.42e-06; eta = 0.119997;  %0.3988

% lambda = 1.24e-05; eta = 0.118811;  %0.4056
% lambda = 3.91e-05; eta = 0.119100;  %0.4020
% lambda = 4.72e-06; eta = 0.119571;  %0.3970

% training
[W, b, cost_train, cost_val] = MiniBatchGD(X_train, Y_train, X_val, Y_val, W, b, lambda, n_epochs, n_batch, eta, m, rho);

% calculate the accuracy
% [X, ~, y] = LoadBatch('test_batch.mat'); % 1

acc = ComputeAccuracy(X_test, y_test, W, b);
disp(acc);

% display the images
% mt = [];
% for i=1:10
%   im = reshape(W(i, :), 32, 32, 3);
%   s_im{i} = (im-min(im(:)))/(max(im(:))-min(im(:)));
%   s_im{i} = permute(s_im{i}, [2, 1, 3]);
%   mt = [mt s_im{i}];
% end
% montage(mt);

% plot the cost function
inds = 1:n_epochs;
plot(inds, cost_train, inds, cost_val);
% disp(cost_train);