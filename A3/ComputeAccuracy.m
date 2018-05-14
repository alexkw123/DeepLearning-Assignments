% ------- ComputeAccuracy.m ---------

function acc = ComputeAccuracy(X, y, W, b, ma)

[~, n] = size(X);
[P, ~, ~, ~, ~, ~] = EvaluateClassifier(X, W, b, ma);
% disp(P);

[~, k] = max(P);
% disp(k);
acc = sum(k' == y)/n;

end