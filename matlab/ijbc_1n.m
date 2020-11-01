function ijbc_1n(gallery_ids_path,gallery_feats_path,probe_ids_path,probe_feats_path,measure)

tic
addpath('../');
addpath('matlab_tools')
addpath_recurse('BLUFR')
%measure = "Hamming";
%+"IJBC1N_" + cfg['backbone_type'] + '_' + str(is_only_arc) + '_' + str(cfg['m']) + 'x' + str(
%        cfg['q'])+
%ijbc_1n("gallery_res50_incev2_ids","gallery_res50_incev2_feats","probe_res50_incev2_ids","probe_res50_incev2_feats")
%ijbc_1n("gallery_res50_xception_ids","gallery_res50_xception_feats","probe_res50_xception_ids","probe_res50_xception_feats")
%ijbc_1n("gallery_incepv2_xception_ids","gallery_incepv2_xception_feats","probe_incepv2_xception_ids","probe_incepv2_xception_feats")
%ijbc_1n("1101gallery_ids","1101gallery_feats","1101probe_ids","1101probe_feats")
disp(gallery_feats_path)
disp(gallery_ids_path)
disp(probe_feats_path)
disp(probe_ids_path)
%gallery_feats = importdata('../data/'+gallery_feats_path+'_G1.csv');
%gallery_ids = importdata('../data/'+gallery_ids_path+'_G1.csv');
%probe_feats = importdata('../data/'+probe_feats_path+'_G1.csv');
%probe_ids = importdata('../data/'+probe_ids_path+'_G1.csv');
%
%
%
%veriFarPoints = [0, kron(10.^(-8:-1), 1:9), 1]; % FAR points for face verification ROC plot
%osiFarPoints = [0, kron(10.^(-4:-1), 1:9), 1]; % FAR points for open-set face identification ROC plot
%rankPoints = [1:10, 20:10:100]; % rank points for open-set face identification CMC plot
%reportVeriFar = 0.001; % the FAR point for verification performance reporting
%reportOsiFar = 0.01; % the FAR point for open-set identification performance reporting
%reportRank = 1; % the rank point for open-set identification performance reporting
%
%numTrials = 1;
%numVeriFarPoints = length(veriFarPoints);
%iom_VR = zeros(numTrials, numVeriFarPoints); % verification rates of the 10 trials
%iom_VR_re = zeros(numTrials, numVeriFarPoints); % verification rates of the 10 trials
%iom_veriFAR = zeros(numTrials, numVeriFarPoints); % verification false accept rates of the 10 trials
%
%numOsiFarPoints = length(osiFarPoints);
%numRanks = length(rankPoints);
%iom_DIR = zeros(numRanks, numOsiFarPoints, numTrials); % detection and identification rates of the 10 trials
%iom_DIR_re = zeros(numRanks, numOsiFarPoints, numTrials); % detection and identification rates of the 10 trials
%iom_osiFAR = zeros(numTrials, numOsiFarPoints); % open-set identification false accept rates of the 10 trials
%
%
%final_dist =pdist2( gallery_feats,probe_feats,measure);
%
%[iom_DIR(:,:,2), iom_osiFAR(2,:)] = OpenSetROC(1-final_dist , gallery_ids, probe_ids, osiFarPoints );
%
%DIR1 =iom_DIR(1,[11 20],2) * 100
%
%%%
%gallery_feats = importdata('../data/'+gallery_feats_path+'_G2.csv');
%gallery_ids = importdata('../data/'+gallery_ids_path+'_G2.csv');
%probe_feats = importdata('../data/'+probe_feats_path+'_G2.csv');
%probe_ids = importdata('../data/'+probe_ids_path+'_G2.csv');
%
%
%
%veriFarPoints = [0, kron(10.^(-8:-1), 1:9), 1]; % FAR points for face verification ROC plot
%osiFarPoints = [0, kron(10.^(-4:-1), 1:9), 1]; % FAR points for open-set face identification ROC plot
%rankPoints = [1:10, 20:10:100]; % rank points for open-set face identification CMC plot
%reportVeriFar = 0.001; % the FAR point for verification performance reporting
%reportOsiFar = 0.01; % the FAR point for open-set identification performance reporting
%reportRank = 1; % the rank point for open-set identification performance reporting
%
%numTrials = 1;
%numVeriFarPoints = length(veriFarPoints);
%iom_VR = zeros(numTrials, numVeriFarPoints); % verification rates of the 10 trials
%iom_VR_re = zeros(numTrials, numVeriFarPoints); % verification rates of the 10 trials
%iom_veriFAR = zeros(numTrials, numVeriFarPoints); % verification false accept rates of the 10 trials
%
%numOsiFarPoints = length(osiFarPoints);
%numRanks = length(rankPoints);
%iom_DIR = zeros(numRanks, numOsiFarPoints, numTrials); % detection and identification rates of the 10 trials
%iom_DIR_re = zeros(numRanks, numOsiFarPoints, numTrials); % detection and identification rates of the 10 trials
%iom_osiFAR = zeros(numTrials, numOsiFarPoints); % open-set identification false accept rates of the 10 trials
%
%
%final_dist =pdist2( gallery_feats,probe_feats,measure);
%
%[iom_DIR(:,:,2), iom_osiFAR(2,:)] = OpenSetROC(1-final_dist , gallery_ids, probe_ids, osiFarPoints );
%
%
%DIR2 =iom_DIR(1,[11 20],2) * 100


%%
gallery_feats = importdata('../data/'+gallery_feats_path+'_G1G2.csv');
gallery_ids = importdata('../data/'+gallery_ids_path+'_G1G2.csv');
probe_feats = importdata('../data/'+probe_feats_path+'_G1G2.csv');
probe_ids = importdata('../data/'+probe_ids_path+'_G1G2.csv');



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

final_dist =pdist2( gallery_feats,probe_feats,measure);
[iom_max_rank,iom_rec_rates] = CMC(1-final_dist',probe_ids,gallery_ids);
%[iom_DIR(:,:,2), iom_osiFAR(2,:)] = OpenSetROC(1-final_dist , gallery_ids, probe_ids, osiFarPoints );
[iom_VR(1,:), iom_veriFAR(1,:)] = EvalROC(1-final_dist, gallery_ids, gallery_ids, veriFarPoints);


%DIR3 =iom_DIR(1,[11 20],2) * 100;

IR_1 = iom_rec_rates(1)* 100;
VR = iom_VR(1,[29 38 56])* 100;
%% DIR3 should equal IR_1
%[DIR1 DIR2 IR_1 VR]
toc
end


