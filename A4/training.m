% ------- training.m ---------

% read in the data
book_fname = 'Datasets/goblet_book.txt';
fid = fopen(book_fname, 'r', 'n','UTF-8');
book_data = fscanf(fid, '%c');
fclose(fid);

% get unique characters
book_chars = unique(book_data);
K = length(book_chars);   % the dimensionality of the output (input) vector of your RNN

char_to_ind = containers.Map('KeyType','char','ValueType','int32'); 
ind_to_char = containers.Map('KeyType','int32','ValueType','char');

for i = 1:K
    char_to_ind(book_chars(i)) = i;
    ind_to_char(i) = book_chars(i);
end

% set hyper-parameters

m = 100; % hidden state size
eta = 0.1;
seq_length = 25;
sig = 0.01;

RNN.b = zeros(m, 1);
RNN.c = zeros(K, 1);
RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;

[~, N] = size(book_data);
X = zeros(K, N);
for i=1:N
  X(char_to_ind(book_data(i)), i) = 1;
end

epoches = 10;
[aRNN, sloss] = MiniBatchGD(X, length(book_data), RNN, m, seq_length, ind_to_char, epoches, eta);

% the last part
h0 = zeros(m, 1);
X_out = X(:, 1:1000);
Y_pre = SynthesizeText(aRNN, hprev, X_out);
chars = blanks(1000);
for i = 1:1000
    [~, k] = max(Y_pre(:,i));
    chars(i) = ind_to_char(k);
end
disp(chars);