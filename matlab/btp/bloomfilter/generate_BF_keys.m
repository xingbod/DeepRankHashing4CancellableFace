function [key] = generate_BF_keys(opts)


% feat = reshape(feat,[opts.H opts.W]);
% N_BITS_BF = opts.N_BITS_BF ; % parameters for BF extraction 2^10
% N_WORDS_BF = opts.N_WORDS_BF;
% BF_SIZE = power(2, N_BITS_BF);
% N_BF_Y = floor(opts.H/N_BITS_BF);
% N_BF_X = floor(opts.W/N_WORDS_BF);
% N_BLOCKS = N_BF_X * N_BF_Y;
% Define permutation key to provide unlinkability
key = zeros(4, opts.N_BITS_BF * opts.N_BF_X / 2);
for j=1:4
    key(j, :) = randperm(opts.N_BITS_BF * opts.N_BF_X / 2);
end



end

