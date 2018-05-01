% % ----------- testing -------------
% test if the ComputeCost and ComputeGradients functions are right
lambda = 0;
[W, b, K, rho, m] = InitializeParameters(X_train, y_train, 3);
[P, s, h] = EvaluateClassifier(X_train(:, 1), W, b);
% function given by professor
[ngrad_b, ngrad_W] = ComputeGradsNum(X_train(:, 1), Y_train(:, 1), W, b, lambda, 1e-6);
% implemented function
[grad_W, grad_b] = ComputeGradients(X_train(:, 1), Y_train(:, 1), P, s, h, W, lambda, K, m);
% relative error
k = length(grad_W);
error_b = zeros(k,1);
error_W = zeros(k,1);
for i = 1:k
    error_b(i) = norm(grad_b{i} - ngrad_b{i})/max(eps,norm(grad_b{i})+norm(ngrad_b{i}));
    error_W(i) = norm(grad_W{i} - ngrad_W{i})/max(eps,norm(grad_W{i})+norm(ngrad_W{i}));
end

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
X_train = X_train(:,1:1000);
Y_train = Y_train(:,1:1000);
y_train = y_train(1:1000);

% initialize the network
% rng(9001);

% set training parameters
n_epochs=10; n_batch=100;

times = 50;
result = zeros(4, times);
% lmin = -6; lmax = -1; emin = log10(0.003); emax = log10(0.5); % coarse
lmin = log10(4e-6); lmax = log10(4e-1); emin = log10(0.1); emax = log10(0.25); % fine
% lambda = 0.9; eta = 0.01;

for i = 1:times
    l = lmin + (lmax - lmin)*rand(1, 1);
    lambda = 10^l;
    
    e = emin + (emax - emin)*rand(1, 1);
    eta = 10^e;
%     eta = e;
    % training
    [W, b, K, rho, m] = InitializeParameters(X_train, y_train);
    [W, b, cost_train, cost_val] = MiniBatchGD(X_train, Y_train, X_val, Y_val, W, b, lambda, n_epochs, n_batch, eta, m, rho);
    
    acc_on_val = ComputeAccuracy(X_val, y_val, W, b);
    
    best_on_val = min(cost_val);
    result(1, i) = lambda;
    result(2, i) = eta;
    result(3, i) = best_on_val;
    result(4, i) = acc_on_val;
    
    disp(i);
end

% write to file

% disp(cost_train);

fid=fopen('Result.txt','a+');
fprintf(fid,'%g\t%g\t%g\t%g\n',result);
fclose(fid);