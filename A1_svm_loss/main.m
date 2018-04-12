% % ----------- testing -------------
% % test if the ComputeCost and ComputeGradients functions are right
% lambda = 0;
% s1 = EvaluateClassifier(X_train(:, 1:9), W, b);
% cost = ComputeCost(X_train(:, 1:9), Y_train(:, 1:9), W, b, lambda);
% % function given by professor
% % [ngrad_b, ngrad_W] = ComputeGradsNumSlow(X_train(1:100, 1), Y_train(:, 1), W(:, 1:100), b, lambda, 1e-6);
% [ngrad_b, ngrad_W] = ComputeGradsNumSlow(X_train(:, 1:9), Y_train(:, 1:9), W, b, lambda, 1e-6);
% % implemented function
% [grad_W, grad_b] = ComputeGradients(X_train(:, 1:9), Y_train(:, 1:9), W, b, lambda, s1);
% % relative error
% error_b = norm(grad_b - ngrad_b)/max(eps,norm(grad_b)+norm(ngrad_b));
% error_W = norm(grad_W - ngrad_W)/max(eps,norm(grad_W)+norm(ngrad_W));


% ------- training process ---------
% read in training, validataion and test data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

% initialize parameters
[d, N] = size(X_train);
K = length(min(y_train):max(y_train));

rng(1);
mean = 0;
stanDev = 0.01;
W = stanDev*randn(K,d) + mean;
b = stanDev*randn(K,1) + mean;

% set training parameters
% lambda=0.0008; n_epochs=50; n_batch=100; eta=.005;   % 32.38%
% lambda=0.0008; n_epochs=50; n_batch=100; eta=.003;   % 32.67%
% lambda=0.0007; n_epochs=60; n_batch=100; eta=.002;   % 34.17%
% lambda=0.0005; n_epochs=60; n_batch=100; eta=.002;   % 34.28%
% lambda=0.01; n_epochs=100; n_batch=100; eta=.001;   % 36.22%
lambda=0; n_epochs=40; n_batch=100; eta=.01;   % 30.97%

% train and validation argument
cost_train = zeros(n_epochs, 1);
cost_val = zeros(n_epochs, 1);

% training
for i = 1 : n_epochs
    for j = 1 : N/n_batch
        j_start = (j-1) * n_batch + 1;
        j_end = j * n_batch;
        inds = j_start : j_end;
        Xbatch = X_train(:, j_start:j_end);
        Ybatch = Y_train(:, j_start:j_end);
        % get mini-batch
        [W, b] = MiniBatchGD(Xbatch, Ybatch, eta, W, b, lambda);
    end
    
    cost_train(i) = (ComputeCost(X_train, Y_train, W, b, lambda));
    cost_val(i) = ComputeCost(X_val, Y_val, W, b, lambda);
end

% calculate the accuracy
[X, ~, y] = LoadBatch('test_batch.mat'); % 1
acc = ComputeAccuracy(X, y, W, b);
disp(acc);

% display the images
mt = [];
for i=1:10
  im = reshape(W(i, :), 32, 32, 3);
  s_im{i} = (im-min(im(:)))/(max(im(:))-min(im(:)));
  s_im{i} = permute(s_im{i}, [2, 1, 3]);
  mt = [mt s_im{i}];
end
montage(mt);

% plot the cost function
inds = 1:n_epochs;
plot(inds, cost_train, inds, cost_val);
