function [B, U] = compressMDSH(X, SHparam)

%
% Input:
%   X = raw data matrix [Nsamples, Nfeatures]
%   SHparam.nbits = number of bits (nbits do not need to be a multiple of 8).
%   SHparam.sigma = Sigma in graph Laplacian (default=0.1).
%   SHparam.epsilon = Constant to avoid DIV0 problems (default = 0.1).
%   SHparam.outlier = Outlier percentile (default = 2.5%)
%
% Output:
%   B = bits (compacted in 8 bits words)
%   U = value of eigenfunctions (bits in B correspond to U>0)
%  
% Spectral Hashing with extension to numerical eigenfunctions
%  
% Based on:
% 1. Spectral Hashing  
% Y. Weiss, A. Torralba, R. Fergus. 
% Advances in Neural Information Processing Systems, 2008.
%
% 2. Semi-supervised learning in very large image collections
% R. Fergus, Y. Weiss, A. Torralba.
% Advances in Neural Information Processing Systems, 2009.
%   



%%%% Algorithm
[Nsamples Ndim] = size(X);
nbits = SHparam.nbits;

%%%%%%%%
%keyboard
% 1) PCA (using pre-computed mean,pc)
Xc = X - ones(Nsamples,1)*SHparam.data_mean;
X = Xc * SHparam.data_pc; % overwrite original data
npca=size(SHparam.data_pc,2);  
%%% do clipping of data distribution per dimension
for a=1:npca
  q = find(X(:,a)<SHparam.clip_lower(a));
  X(q,a) = SHparam.clip_lower(a);
  q2 = find(X(:,a)>SHparam.clip_upper(a));
  X(q2,a) = SHparam.clip_upper(a);
end

%%%%%%%%
% 2) Do interpolation
for a=1:nbits
  U(:,a) = interp1(SHparam.bins(:,a),SHparam.eigenfunc(:,a),X(:,SHparam.chosen_dim(a)),'linear','extrap');
  U(:,a)=U(:,a)-SHparam.thresholds(a);
end
% extra nonlinear 
if (SHparam.softmod)
    U = 2./(1+exp(-8*sin(U*pi*SHparam.alpha)))-1;
end
% U = 2./(1+exp(-8*sin(U*pi*0.6)))-1;

%%%%%%%%
% 3) Threshold 

B = compactbit(U > 0);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cb = compactbit(b)
%
% b = bits array
% cb = compacted string of bits (using words of 'word' bits)

[nSamples nbits] = size(b);
nwords = ceil(nbits/8);
cb = zeros([nSamples nwords], 'uint8');

for j = 1:nbits
    w = ceil(j/8);
    cb(:,w) = bitset(cb(:,w), mod(j-1,8)+1, b(:,j));
end

