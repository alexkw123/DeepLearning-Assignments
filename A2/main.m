% % ----------- testing -------------
% test if the ComputeCost and ComputeGradients functions are right
lambda = 0;
[P, h, s1] = EvaluateClassifier(X_train(:, 5), W, b);
% function given by professor
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(X_train(:, 5), Y_train(:, 5), W, b, lambda, 1e-6);
% implemented function
[grad_W, grad_b] = ComputeGradients(X_train(:, 5), Y_train(:, 5), P, h, s1, W, lambda, K, m);
% relative error
error_b1 = norm(grad_b{1} - ngrad_b{1})/max(eps,norm(grad_b{1})+norm(ngrad_b{1}));
error_b2 = norm(grad_b{2} - ngrad_b{2})/max(eps,norm(grad_b{2})+norm(ngrad_b{2}));
error_W1 = norm(grad_W{1} - ngrad_W{1})/max(eps,norm(grad_W{1})+norm(ngrad_W{1}));
error_W2 = norm(grad_W{2} - ngrad_W{2})/max(eps,norm(grad_W{2})+norm(ngrad_W{2}));

% ------- training process ---------
% read in training, validataion and test data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

% transform training data to have zero mean
mean_X = mean(X_train, 2);
X_train = X_train - repmat(mean_X, [1, size(X_train, 2)]);
% subtract it from the input vectors in the validation and test sets
X_val = X_val - repmat(mean_X, [1, size(X_val, 2)]);
X_test = X_test - repmat(mean_X, [1, size(X_test, 2)]);

% set up data structure
m = 50;
[d, N] = size(X_train);
[W, b, K] = InitializeParameters(X_train, y_train, m);

% set training parameters
% lambda=0; n_epochs=40; n_batch=100; eta=.1;    % 25.61%
% lambda=0; n_epochs=40; n_batch=100; eta=.01;   % 36.87%
% lambda=.1; n_epochs=40; n_batch=100; eta=.01;  % 33.36%
% lambda=1; n_epochs=40; n_batch=100; eta=.01;   % 21.93%
lambda=0.01; n_epochs=100; n_batch=100; eta=.001; % 37.38%

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
