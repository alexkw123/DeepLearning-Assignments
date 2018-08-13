% ------- training.m ---------

% read in the data
fname = 'Datasets/condensed_2017.json';
fid = fopen(fname, 'r', 'n','UTF-8');
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
data = jsondecode(str);
m = length(data);
tweet_data = "";
for i = 1:m
    tweet_data = strcat(tweet_data,data(i).text);
    tweet_data = strcat(tweet_data,"<");
end
tweet_data = char(tweet_data);

% get unique characters
tweet_chars = unique(tweet_data);
K = length(tweet_chars);   % the dimensionality of the output (input) vector of your RNN

char_to_ind = containers.Map('KeyType','char','ValueType','int32'); 
ind_to_char = containers.Map('KeyType','int32','ValueType','char');

for i = 1:K
    char_to_ind(tweet_chars(i)) = i;
    ind_to_char(i) = tweet_chars(i);
end

% set hyper-parameters

m = 100; % hidden state size
eta = 0.1;
seq_length = 10; % can be played around with
sig = 0.01;

RNN.b = zeros(m, 1);
RNN.c = zeros(K, 1);
RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;

[~, N] = size(tweet_data);
X = zeros(K, N);
for i=1:N
  X(char_to_ind(tweet_data(i)), i) = 1;
end

epoches = 7;
[aRNN, sloss] = MiniBatchGD(X, length(tweet_data), RNN, m, seq_length, ind_to_char, epoches, eta, char_to_ind);

% the last part
h0 = zeros(m, 1);
X_out = X(:, 1:1000);
Y_pre = SynthesizeText(aRNN, h0, X_out);
chars = blanks(1000);
for i = 1:1000
    [~, k] = max(Y_pre(:,i));
    chars(i) = ind_to_char(k);
end
disp(chars);