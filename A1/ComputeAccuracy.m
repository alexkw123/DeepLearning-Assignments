function acc = ComputeAccuracy(X, y, W, b)

[~, n] = size(X);
P = EvaluateClassifier(X, W, b);
disp(P);

[~, k] = max(P);
disp(k);
acc = sum(k' == y)/n;

end