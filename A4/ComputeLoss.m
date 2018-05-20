function J = ComputeLoss(X, Y, RNN, h0)

[P, ~] = ForwardPass(RNN, h0, X);
J = -sum(log(sum(Y.*P, 1)));

end