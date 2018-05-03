function g = BatchNormBackPass(g, s, mu, v)
% s, mu, v: cell array
[~, n] = size(g{1});
k = length(g); % number of layers
% g = cell2mat(g);
dv = (-0.5)*g{1}*diag(v.^(-1.5))*diag(s(1)-mu);
dmu = -g{1}*diag(v.^(-0.5));
for i=2:k
    dv = dv-0.5*g{i}*diag(v.^(-1.5))*diag(s(i)-mu);
    dmu = dmu - g{i}*diag(v.^(-0.5));
end
% ng = cell(k,1);
for i=1:k
    g{i} = g{i}*diag(v.^(-0.5)) + 2/n*dv*diag(s(i)-mu) + dmu/n;
end
end