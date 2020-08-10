% clear all;
% close all;
% 
% load('data\bloomfilter\iriscode\labels.mat')
% % load('data\bloomfilter\iriscode\templates.mat')
% load('data\bloomfilter\iriscode\Codes_lg_templates.mat')
% % load('data\bloomfilter\iriscode\Codes_lg_labels.mat')
% % load('data\bloomfilter\iriscode\Codes_qsw_templates.mat')

addpath('matlab_tools');
addpath_recurse("btp")

% === To use this code ===
% 1. Select input parameters value for (tau,P,K, m)
% 2. Generate random permutation matrix (e.g. tokens) for IFO hashing: [ PermuteMatx ] = Generate_PermMx( InputSample,P,m)
% 3. Generate IFO hashed code: [IFOCode,PermuteMatx] = IFO(InputSample,tau,PermuteMatx,P, K,m)
% 4. Matching between different IFO hashed codes:  [Score, No_of_Collision] = IFO_matching(IFOcode1, IFOcode2)


tau=0;P=3;K=100; m=800;
templates =  logical(templates);
[ PermuteMatx ] = Generate_PermMx(templates(1,:),P,m);
[IFOCode,PermuteMatx] = IFO(templates,tau,PermuteMatx,P, K,m);

% 

for i=1:size(IFOCode,1)
    i
    for j=i:size(IFOCode,1)
        [max_score, No_of_Collision] = IFO_matching(IFOCode(i,:), IFOCode(j,:))
        score(i,j)=max_score;
    end
end

% chaneg a matching protocol
% scores=1-pdist2(IFOCode,IFOCode,'Hamming');
hamming_gen_score = scores(labels'==labels);
hamming_gen_score = hamming_gen_score(find(hamming_gen_score~=1));
hamming_gen_score = hamming_gen_score(find(hamming_gen_score~=0));
hamming_imp_score = scores(labels'~=labels);
hamming_imp_score = hamming_imp_score(find(hamming_imp_score~=0));


[EER_HASH_orig, mTSR, mFAR, mFRR, mGAR,threshold] =computeperformance(hamming_gen_score, hamming_imp_score, 0.001);  % isnightface 3.43 % 4.40 %
[FAR_orig,FRR_orig] = FARatThreshold(hamming_gen_score,hamming_imp_score,threshold);

