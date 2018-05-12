function g = BatchNormBackPass(g, s, mu, v, eps)
% s, mu, v: cell array
[~, n] = size(g{1});
k = length(g); % number of batches
v = v+eps;
dv = g{1}*diag(v.^(-1.5))*diag(s(:, 1)-mu);
dmu = g{1}*diag(v.^(-0.5));
for i=2:k
    dv = dv + g{i}*diag(v.^(-1.5))*diag(s(:, i)-mu);
    dmu = dmu + g{i}*diag(v.^(-0.5));
end
dv = -0.5*dv;
dmu = -dmu;
for i=1:k
    g{i} = g{i}*diag(v.^(-0.5)) + 2/n*dv*diag(s(:, i)-mu) + dmu/n;
end
end