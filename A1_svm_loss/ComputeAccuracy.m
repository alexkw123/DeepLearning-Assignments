function acc = ComputeAccuracy(X, y, W, b)

[~, n] = size(X);
P = EvaluateClassifier(X, W, b);

[~, k] = max(P);
acc = sum(k' == y)/n;

end