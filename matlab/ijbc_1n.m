addpath('../');
addpath('matlab_tools')
addpath_recurse('BLUFR')
addpath_recurse('btp')
addpath('k_reciprocal_re_ranking')

%%
k1 = 20;
k2 = 6;
lambda = 0.3;
measure = 'Hamming';
% ds = "LFW";
log_path = "logs/log_ijbc_idnew.log";

%%
Descriptor_G1 = importdata("../embedding_dl/Insight_ResNet100_ijbc_G1_feat_dlIoM_512x8.csv");
Descriptor_G2 = importdata("../embedding_dl/Insight_ResNet100_ijbc_G2_feat_dlIoM_512x8.csv");
Descriptor_P = importdata("../embedding_dl/Insight_ResNet100_ijbc_P_feat_dlIoM_512x8.csv");

Names_G1 = importdata("../embedding_dl/Insight_ResNet100_ijbc_G1_name_dl_512x8.txt");
Names_G2 = importdata("../embedding_dl/Insight_ResNet100_ijbc_G2_name_dl_512x8.txt");
Names_P = importdata("../embedding_dl/Insight_ResNet100_ijbc_P_name_dl_512x8.txt");

Descriptor_G = [Descriptor_G1 ;Descriptor_G2];
Names_G =[Names_G1; Names_G2];

final_dist_o2 =(pdist2( Descriptor_G,Descriptor_P,  measure));

final_dist_o2_re = re_ranking_score(final_dist_o2',Names_G,Names_P,Descriptor_G,Descriptor_P, k1, k2, lambda,measure);




veriFarPoints = [0, kron(10.^(-8:-1), 1:9), 1]; % FAR points for face verification ROC plot
osiFarPoints = [0, kron(10.^(-4:-1), 1:9), 1]; % FAR points for open-set face identification ROC plot
rankPoints = [1:10, 20:10:100]; % rank points for open-set face identification CMC plot
reportVeriFar = 0.001; % the FAR point for verification performance reporting
reportOsiFar = 0.01; % the FAR point for open-set identification performance reporting
reportRank = 1; % the rank point for open-set identification performance reporting

numTrials = 1;
numVeriFarPoints = length(veriFarPoints);
iom_VR = zeros(numTrials, numVeriFarPoints); % verification rates of the 10 trials
iom_VR_re = zeros(numTrials, numVeriFarPoints); % verification rates of the 10 trials
iom_veriFAR = zeros(numTrials, numVeriFarPoints); % verification false accept rates of the 10 trials

numOsiFarPoints = length(osiFarPoints);
numRanks = length(rankPoints);
iom_DIR = zeros(numRanks, numOsiFarPoints, numTrials); % detection and identification rates of the 10 trials
iom_DIR_re = zeros(numRanks, numOsiFarPoints, numTrials); % detection and identification rates of the 10 trials
iom_osiFAR = zeros(numTrials, numOsiFarPoints); % open-set identification false accept rates of the 10 trials

%% Compute the cosine similarity score between the test samples.

[iom_DIR(:,:,2), iom_osiFAR(2,:)] = OpenSetROC(1-final_dist_o2, Names_G, Names_P, osiFarPoints );

[iom_DIR_re(:,:,2), iom_osiFAR(2,:)] = OpenSetROC(1-final_dist_o2_re, Names_G, Names_P, osiFarPoints );


