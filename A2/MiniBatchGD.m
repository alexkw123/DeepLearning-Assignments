function [W, b, cost_train, cost_val] = MiniBatchGD(X_train, Y_train, X_val, Y_val, W, b, lambda, n_epochs, n_batch, eta, m, rho)
[d, N] = size(X_train);
[K, ~] = size(Y_train);

% train and validation argument
cost_train = zeros(n_epochs, 1);
cost_val = zeros(n_epochs, 1);

original_training_cost = ComputeCost(X_train, Y_train, W, b, lambda);

% keep a record of the best model
% best_W = W;
% best_b = b;
% best_val_cost = 10;

% training
for i = 1 : n_epochs
    % initialize momentum
    v_W1 = zeros(m, d);
    v_b1 = zeros(m, 1);
    v_W2 = zeros(K, m);
    v_b2 = zeros(K, 1);
    
    for j = 1 : N/n_batch
    j_start = (j-1) * n_batch + 1;
    j_end = j * n_batch;
    inds = j_start : j_end;
    Xbatch = X_train(:, inds);
    Ybatch = Y_train(:, inds);
    % compute gradient
    [P, h, s1] = EvaluateClassifier(Xbatch, W, b);
    [K, ~] = size(Ybatch);
    [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, h, s1, W, lambda, K, m);
    % momentum
    v_W1 = rho * v_W1 + eta * grad_W{1};
    v_b1 = rho * v_b1 + eta * grad_b{1};
    v_W2 = rho * v_W2 + eta * grad_W{2};
    v_b2 = rho * v_b2 + eta * grad_b{2};
    % update the weights and the bias.
    W{1} = W{1} - v_W1;
    b{1} = b{1} - v_b1;
    W{2} = W{2} - v_W2;
    b{2} = b{2} - v_b2;
    end
    
    cost_train(i) = ComputeCost(X_train, Y_train, W, b, lambda);
    cost_val(i) = ComputeCost(X_val, Y_val, W, b, lambda);
    % abort when training cost is too large
    if cost_train(i) > 3 * original_training_cost
        break;
    end
    
    % record best model
%     if cost_val(i) < best_val_cost
%         best_W = W;
%         best_b = b;
%         best_val_cost = cost_val(i);
%     end
    
    % decay learning rate by 0.95
%     if mod(i, 10) == 0
%         eta = 0.1 * eta;
%     end
    eta = 0.8 * eta;
end