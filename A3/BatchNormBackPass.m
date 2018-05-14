% ------- BatchNormBackPass.m ---------

function g = BatchNormBackPass(g, s, mu, v)
% s, mu, v: cell array
n = length(g); % number of batches
v = v+eps;
dv = zeros(size(g{1}));
dmu = zeros(size(g{1}));
for i=1:n
    dv = dv + g{i}*diag(v.^(-1.5))*diag(s(:, i)-mu);
    dmu = dmu + g{i}*diag(v.^(-0.5));
end
dv = -0.5*dv;
dmu = -dmu;
for i=1:n
    g{i} = g{i}*diag(v.^(-0.5)) + 2/n*dv*diag(s(:, i)-mu) + dmu/n;
end
end