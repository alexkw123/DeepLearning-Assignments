% ------- CoarseFineSearch.m ---------

% read in training, validataion and test data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

% transform training data to have zero mean
% mean_X = mean(X_train, 2);
% X_train = X_train - repmat(mean_X, [1, size(X_train, 2)]);
% % subtract it from the input vectors in the validation and test sets
% X_val = X_val - repmat(mean_X, [1, size(X_val, 2)]);
% X_test = X_test - repmat(mean_X, [1, size(X_test, 2)]);

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
% lmin = -6; lmax = -1; emin = log10(0.003); emax = log10(0.04); % coarse
lmin = log10(4e-6); lmax = log10(4e-1); emin = log10(0.001); emax = log10(0.003); % fine
% lambda = 0.9; eta = 0.01;

% initialize
k = 3;  % layers
hnodes = [50,30];
[W, b, K, rho] = InitializeParameters(X_train, y_train, k, hnodes);

for i = 1:times
    l = lmin + (lmax - lmin)*rand(1, 1);
    lambda = 10^l;
    
    e = emin + (emax - emin)*rand(1, 1);
    eta = 10^e;
    
    [W, b, cost_train, cost_val, ma] = MiniBatchGD(X_train, Y_train, X_val, Y_val, W, b, lambda, n_epochs, n_batch, eta, rho);
    
    acc_on_val = ComputeAccuracy(X_val, y_val, W, b, ma);
    
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