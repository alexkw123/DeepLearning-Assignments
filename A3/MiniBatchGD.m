function [W, b, cost_train, cost_val] = MiniBatchGD(X_train, Y_train, X_val, Y_val, W, b, lambda, n_epochs, n_batch, eta, rho)

[~, N] = size(X_train);
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
    layers = length(W);
    v_W = cell(layers);
    v_b = cell(layers);
    for l = 1 : layers
        v_W{l} = zeros(size(W{l}));
        v_b{l} = zeros(size(b{l}));
    end
    
    for j = 1 : N/n_batch
        j_start = (j-1) * n_batch + 1;
        j_end = j * n_batch;
        inds = j_start : j_end;
        Xbatch = X_train(:, inds);
        Ybatch = Y_train(:, inds);
        % compute gradient
        [P, s, h] = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, s, h, W, lambda, b);
        % momentum
        for l = 1 : layers
            v_W{l} = rho * v_W{l} + eta * grad_W{l};
            v_b{l} = rho * v_b{l} + eta * grad_b{l};
            % update the weights and the bias.
            W{l} = W{l} - v_W{l};
            b{l} = b{l} - v_b{l};
        end
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
    eta = 0.9 * eta;
end