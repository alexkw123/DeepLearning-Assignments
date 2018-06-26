% ------- MiniBatchGD.m ---------

function [RNN, sloss] = MiniBatchGD(X, N, RNN, m, seq_length, ind_to_char, epoches, eta)

it = 1; % number of iterations
sloss = zeros(round(epoches*(N-seq_length)/seq_length),1);

% e: keeps track of where in the book you are
for f = fieldnames(RNN)'
    M.(f{1}) = zeros(size(RNN.(f{1})));
end

for epoch = 1:epoches
    e = 1;
    hprev = zeros(m, 1);
    
    while(e <= N-seq_length-1)
        X_batch = X(:, e:e+seq_length-1);
        Y_batch = X(:, e+1:e+seq_length);

        [P, H, loss] = ForwardPass(RNN, hprev, X_batch, Y_batch);
        grads = ComputeGradients(X_batch, Y_batch, RNN, P, H, hprev);
        
        % clip gradients to avoid the exploding gradient problem
        for f = fieldnames(grads)'
            grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
        end
        % set hprev to last computed hidden state by forward pass
%         h0 = hprev;
        hprev = H(:, end);

        % AdaGrad
        for f = fieldnames(RNN)'
            M.(f{1}) = M.(f{1}) + grads.(f{1}).^2;
            RNN.(f{1}) = RNN.(f{1}) - eta * grads.(f{1})./sqrt(M.(f{1})+eps);
        end

        if it == 1  % at the beginning, store the original loss
            smooth_loss = loss;
        else
            smooth_loss = .999* smooth_loss + .001 * loss;
            sloss(it) = smooth_loss;
        end

        e = e + seq_length;
        
%         if (or (it == 1, it == 1000))
%             disp(smooth_loss);
%         end
%         
%         if (it == 4000)
%             disp(smooth_loss);
%         end
        
        if (or(mod(it, 10000) == 0, it==1))
            disp(smooth_loss);
            X_out = X(:, e:e+199);
            Y_pre = SynthesizeText(RNN, hprev, X_out);
            chars = blanks(200);
            for i = 1:200
                [~, k] = max(Y_pre(:,i));
                chars(i) = ind_to_char(k);
            end
            disp(chars);
            disp(it);
        end
        
        it = it + 1;
    end
end