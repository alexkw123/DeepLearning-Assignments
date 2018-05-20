
% read in the data
book_fname = 'Datasets/goblet_book.txt';
fid = fopen(book_fname, 'r');
book_data = fscanf(fid, '%c');
fclose(fid);

% get unique characters
book_chars = unique(book_data);
RNN.K = length(book_chars);   % the dimensionality of the output (input) vector of your RNN

RNN.char_to_ind = containers.Map('KeyType','char','ValueType','int32'); 
RNN.ind_to_char = containers.Map('KeyType','int32','ValueType','char');

for i = 1:K
    RNN.char_to_ind(book_chars(i)) = i;
    RNN.ind_to_char(i) = book_chars(i);
end

% set hyper-parameters

RNN.m = 100; % hidden state size
RNN.eta = 0.1;
RNN.seq_length = 25;

RNN.b = zeros(m, 1);
RNN.c = zeros(K, 1);
% weight matrices
sig = 0.01;
RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;

[~, RNN.N] = size(book_data);
X = zeros(RNN.K, RNN.N);
for i=1:RNN.N
  X(RNN.char_to_ind(book_data(i)), i) = 1;
end

h0 = zeros(RNN.m, 1);
Y = SynthesizeText(RNN, h0, X(:, 1:5));

X_chars = book_data(1:RNN.seq_length);
Y_chars = book_data(2:RNN.seq_length+1);