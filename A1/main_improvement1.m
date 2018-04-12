% ------- training process ---------
% read in training, validataion and test data
% 0.4030
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

[d, N] = size(X_train);
K = 10;

% [X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
% [X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');

[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

% initialize parameters
mean = 0;
stanDev = 0.01;
% W = stanDev*randn(K,d) + mean;
% Xavier initialization
W = 1/sqrt(d)*ones(K,d);
% W = stanDev*randn(K,d) + mean;
b = stanDev*randn(K,1) + mean;

% set training parameters
% lambda=0; n_epochs=40; n_batch=100; eta=.1;    % 25.61%
lambda=0.01; n_epochs=100; n_batch=100; eta=.01;   % 36.87%
% lambda=.1; n_epochs=40; n_batch=100; eta=.01;  % 33.36%
% lambda=1; n_epochs=40; n_batch=100; eta=.01;   % 21.93%

% train and validation argument
cost_train = zeros(n_epochs, 1);
cost_val = zeros(n_epochs, 1);

best_W = W;
best_b = b;
best_model_cost = Inf;
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
    
    % early stop
    if i ~= 1 && cost_val(i) < best_model_cost
        best_model_cost = cost_val(i);
        best_W = W;
        best_b = b;
    end
    
    % decay learning rate by 0.9
    if mod(i, 10) == 0
        eta = 0.9 * eta;
        disp(eta);
    end
end

% calculate the accuracy
[X, ~, y] = LoadBatch('test_batch.mat'); % 1
acc = ComputeAccuracy(X, y, best_W, best_b);
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
legend('train','validation');
