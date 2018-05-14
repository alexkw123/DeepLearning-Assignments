% ------- gradientCheck.m ---------
% test if the ComputeCost and ComputeGradients functions are right
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

lambda = 0;
k = 3;  % layers
hnodes = [50,30];
[W, b, K, rho] = InitializeParameters(X_train, y_train, k, hnodes);

% function given by professor
[ngrad_b, ngrad_W] = ComputeGradsNum(X_train(:, 1:5), Y_train(:, 1:5), W, b, lambda, 1e-6);
% implemented function
[grad_W, grad_b] = ComputeGradients(X_train(:, 1:5), Y_train(:, 1:5), W, lambda, b);
% relative error
error_b = zeros(k,1);
error_W = zeros(k,1);
for i = 1:k
    error_b(i) = norm(grad_b{i} - ngrad_b{i})/max(norm(grad_b{i}),norm(ngrad_b{i}));
    error_W(i) = norm(grad_W{i} - ngrad_W{i})/max(norm(grad_W{i}),norm(ngrad_W{i}));
end
disp(error_W);
disp(error_b);