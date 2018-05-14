% ------- MiniBatchGD.m ---------

function [W, b, cost_train, cost_val, ma] = MiniBatchGD(X_train, Y_train, X_val, Y_val, W, b, lambda, n_epochs, n_batch, eta, rho)

[~, N] = size(X_train);
% train and validation argument
cost_train = zeros(n_epochs, 1);
cost_val = zeros(n_epochs, 1);
alpha = 0.99;

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
        [P, s, sp, h, mu, v] = EvaluateClassifier(Xbatch, W, b);
        % compute gradient
        if j==1
            ma.mu = mu;
            ma.v = v;
        else
            for e = 1:length(mu)
                ma.mu{e} = alpha*ma.mu{e} + (1-alpha)*mu{e};
                ma.v{e} = alpha*ma.v{e} + (1-alpha)*v{e};
            end
        end
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, s, h, W, lambda, b, sp, mu, v);
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