function [perm_feat] = add_unlinkability(features, keyPERM,opts)
%ADD_UNLINKABILITY Summary of this function goes here
%'''Permutes rows within regions of an iris-code to achieve unlinkability'''
N_BITS_BF = opts.N_BITS_BF;
N_WORDS_BF = opts.N_WORDS_BF;
N_BF_X = opts.N_BF_X;
features = reshape(features,[opts.H opts.W]);

% divide iris-code in four regions, and reshape each region to a size [N_BITS_BF * N_BF_X / 2, N_WORDS_BF]
featsAux1 = reshape(features(1 : N_BITS_BF, 1 : N_BF_X / 2 * N_WORDS_BF), [N_BITS_BF * N_BF_X / 2, N_WORDS_BF]);
featsAux2 = reshape(features(1 : N_BITS_BF, N_BF_X / 2 * N_WORDS_BF+1 : N_BF_X * N_WORDS_BF), [(N_BITS_BF * N_BF_X / 2), N_WORDS_BF]);
featsAux3 = reshape(features(N_BITS_BF+1 : 2*N_BITS_BF, 1 : (N_BF_X / 2) * N_WORDS_BF), [(N_BITS_BF * N_BF_X / 2), N_WORDS_BF]);
featsAux4 = reshape(features(N_BITS_BF+1 : 2*N_BITS_BF, (N_BF_X / 2 * N_WORDS_BF)+1 : N_BF_X * N_WORDS_BF), [(N_BITS_BF * N_BF_X / 2), N_WORDS_BF]);

% permute rows within each region
perm_feat(1: N_BITS_BF, 1:(N_BF_X / 2 * N_WORDS_BF)) = reshape(featsAux1(keyPERM(1, :), :), [N_BITS_BF, N_BF_X / 2 * N_WORDS_BF]);
perm_feat(1: N_BITS_BF, (N_BF_X / 2 * N_WORDS_BF)+1: N_BF_X * N_WORDS_BF) = reshape(featsAux2(keyPERM(2, :), :), [N_BITS_BF, (N_BF_X / 2) * N_WORDS_BF]);
perm_feat(N_BITS_BF+1: 2 * N_BITS_BF, 1: (N_BF_X / 2) * N_WORDS_BF) = reshape(featsAux3(keyPERM(3, :), :), [N_BITS_BF, (N_BF_X / 2) * N_WORDS_BF]);
perm_feat(N_BITS_BF+1: 2 * N_BITS_BF, (N_BF_X / 2 * N_WORDS_BF)+1: N_BF_X * N_WORDS_BF) = reshape(featsAux4(keyPERM(4, :), :), [N_BITS_BF, (N_BF_X / 2) * N_WORDS_BF]);



end

