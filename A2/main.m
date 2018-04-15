% % ----------- testing -------------
% % test if the ComputeCost and ComputeGradients functions are right
% lambda = 0;
% [P, h, s1] = EvaluateClassifier(X_train(:, 1:5), W, b);
% % function given by professor
% [ngrad_b, ngrad_W] = ComputeGradsNum(X_train(:, 1:5), Y_train(:, 1:5), W, b, lambda, 1e-6);
% % implemented function
% [grad_W, grad_b] = ComputeGradients(X_train(:, 1:5), Y_train(:, 1:5), P, h, s1, W, lambda, K, m);
% % relative error
% error_b1 = norm(grad_b{1} - ngrad_b{1})/max(eps,norm(grad_b{1})+norm(ngrad_b{1}));
% error_b2 = norm(grad_b{2} - ngrad_b{2})/max(eps,norm(grad_b{2})+norm(ngrad_b{2}));
% error_W1 = norm(grad_W{1} - ngrad_W{1})/max(eps,norm(grad_W{1})+norm(ngrad_W{1}));
% error_W2 = norm(grad_W{2} - ngrad_W{2})/max(eps,norm(grad_W{2})+norm(ngrad_W{2}));

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

% only use 100 examples
X_train = X_train(:,1:100);
Y_train = Y_train(:,1:100);
y_train = y_train(1:100);

% initialize the network
% rng(9001);
[W, b, K, rho, m] = InitializeParameters(X_train, y_train);

% set training parameters
n_epochs=200; n_batch=10;

times = 1;
result = zeros(3, times);
lmin = -6; lmax = -2; emin = -3; emax = -1;

for i = 1:times
%     l = lmin + (lmax - lmin)*rand(1, 1);
    lambda = 0;
    
%     e = emin + (emax - emin)*rand(1, 1);
    eta = 0.01;

    % training
    [W, b, cost_train, cost_val] = MiniBatchGD(X_train, Y_train, X_val, Y_val, W, b, lambda, n_epochs, n_batch, eta, m, rho);

    best_on_val = min(cost_val);
    result(1, i) = lambda;
    result(2, i) = eta;
    result(3, i) = best_on_val;
    
    disp(i);
end

% write to file

disp(result);

% fid=fopen('Result.txt','a+');
% fprintf(fid,'%g\t %g\t %g\n',result);
% fclose(fid);

% calculate the accuracy
% [X, ~, y] = LoadBatch('test_batch.mat'); % 1
% acc = ComputeAccuracy(X, y, W, b);
% disp(acc);

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
disp(cost_train);