

function SHparam = trainSH_numerical(X, SHparam)

%
% Input:
%   X = raw data matrix [Nsamples, Nfeatures]
%   SHparam.nbits = number of bits (nbits do not need to be a multiple of 8).
%   SHparam.sigma = Sigma in graph Laplacian (default=0.1).
%   SHparam.epsilon = Constant to avoid DIV0 problems (default = 0.1).
%   SHparam.outlier = Outlier percentile (default = 2.5%)
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
% 3. Multidimensional Spectral Hashing
% Y. Weiss, A. Torralba, R. Fergus. 
% ECCV 2012.



  
%%%% Defaults  
if ~isfield(SHparam,'sigma')
  SHparam.sigma = 0.1;
end

if ~isfield(SHparam,'epsilon')
  SHparam.epsilon = 0.1;
end

if ~isfield(SHparam,'outlier')
  SHparam.outlier = 2.5;
end

if ~isfield(SHparam,'min_weight')
  SHparam.min_weight = 0.1;
end

if ~isfield(SHparam,'nbuckets')
    SHparam.nbuckets=SHparam.nbits;
end

if ~isfield(SHparam,'doPCA')
    SHparam.doPCA=1;
end

%%%% Algorithm
[Nsamples Ndim] = size(X);
nbits = SHparam.nbits;

%%%%%%%%
%keyboard
% 1) PCA

npca=Ndim;

if (SHparam.doPCA==1)
    mu = mean(X,1);
    Xc = X - ones(Nsamples,1)*mu;
    %npca = min(nbits, Ndim);
    %[pc, l] = eigs(cov(Xc), npca);
    [pc,ss,vv]=svd(cov(Xc));
    X = Xc * pc; % overwrite original data

%% save PCA parameters
    SHparam.data_mean = mu;
    SHparam.data_pc   = pc;
else
    SHparam.data_pc=eye(Ndim);
    SHparam.data_mean = 0*mean(X,1);
end

% 1.5) Remove outliers
clip_lower = prctile(X,SHparam.outlier);
clip_upper = prctile(X,100-SHparam.outlier);
  
%%% do clipping of data distribution per dimension
for a=1:npca
  q = find(X(:,a)<clip_lower(a));
  X(q,a) = clip_lower(a);
  q2 = find(X(:,a)>clip_upper(a));
  X(q2,a) = clip_upper(a);
end

%% save clipping thresholds 
SHparam.clip_lower = clip_lower;
SHparam.clip_upper = clip_upper;

%%%%%%%%
  
% 2) Compute numerical eigenfunctions per dimension 
for a=1:npca
  [bins(:,a),yys(:,:,a),lambdas(:,a),pp]=numericalEigenFunctions2(X(:,a),SHparam.sigma,SHparam.epsilon);
end

%%%%%%%%

% remove dimensions with numerical errors
mmm=max(lambdas);
Inotvalid=find(mmm<0.999); % max eigenvalue should be one along each dim
lambdas(:,Inotvalid)=1e10; % this will never get chosen

% 3) Select eigenfunctions based on eigenvector size


all_l = lambdas(:);
% remove trivial ones
q=find(all_l<1e-10);
all_l(q) = 1e10;

all_w=1-all_l; % weights for affinity are 1-eval
all_w=all_w/max(all_w);

nAboveT=length(find(all_w>SHparam.min_weight));
[lambda_out,ind] = sort(all_w,'descend');

% evaluate first nAboveT eigenvectors to find
% thresholds
smallN=min(nAboveT,nbits);
[ii,jj]=ind2sub(size(lambdas),ind(1:smallN));
for a=1:smallN
    uuSmall(:,a) = yys(:,ii(a),jj(a));
end

if (nAboveT>nbits)
    use_l = ind(1:nbits);
    use_T=ones(nbits,1)*prctile(uuSmall(:),50);
    weights=all_w(ind(1:nbits));
else
    numberRep=floor(nbits/nAboveT);
    pct=100*(1:numberRep+1)/(numberRep+2);% percentiles for the thresholds
    
    % always have at least one thresh at 50%
    I=find(pct==50);
    if (length(I)==0)
        pct(1)=50;
    end
   
    
    % if you have one repetition the threshold is at 50th percentile
    % if you have two, it is at 33% and 66 etc.
    use_l=ind(1:nAboveT);use_T=ones(nAboveT,1)*prctile(uuSmall(:),pct(1));
    for rr=1:numberRep-1
        use_l=[use_l;ind(1:nAboveT)];
        use_T=[use_T;ones(nAboveT,1)*prctile(uuSmall(:),pct(rr+1))];
    end
    % make sure we end up with exactly nbits in the end
    nLeft=nbits-length(use_l);
    use_l=[use_l;ind(1:nLeft)];
    use_T=[use_T;ones(nLeft,1)*prctile(uuSmall(:),pct(end))];
    weights=all_w(use_l);
end

%%% get indices of picked eigenvectors
[ii,jj]=ind2sub(size(lambdas),use_l);

%%%%%%%%

% 4) now select the eigenfunctions

for a=1:nbits

  %% get bin indices
  bins_out(:,a) = bins(:,jj(a));
  %% and function value
  uu(:,a) = yys(:,ii(a),jj(a));
  lambda_out(a)=lambdas(ii(a),jj(a));
end

SHparam.bins = bins_out;
SHparam.eigenfunc = uu;
SHparam.eigenval = lambda_out;
SHparam.chosen_dim = jj;
% for compatability with trainSH outputs:
SHparam.lambdas=weights;
SHparam.modes=zeros(nbits,Ndim);
for i=1:nbits
    SHparam.modes(i,jj(i))=1;
end
SHparam.thresholds=use_T;



% SHparam.deltas=[ones(1,nbits);findMDSHdeltas(SHparam,SHparam.nbuckets)];
