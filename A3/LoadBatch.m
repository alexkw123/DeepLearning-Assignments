% ------- LoadBatch.m ---------

function [X, Y, y] = LoadBatch(filename)
% the function to read in the data from the file and return the image and
% label data in seperate files
A = load(filename);
X = im2double(A.data');
y = A.labels;

% encode the labels between 1-10
for i = 1:length(y)
    y(i) = y(i) + 1;
end

% calculate the one-hot representation of the label
K = length(min(y):max(y));
N = length(y);
Y = zeros(K, N);
for i = 1:K
    rows = y == i;
    Y(i, rows) = 1;
end

end