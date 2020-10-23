function ijbc_1n(gallery_ids_path,gallery_feats_path,probe_ids_path,probe_feats_path)

addpath('../');
addpath('matlab_tools')
addpath_recurse('BLUFR')

%+"IJBC1N_" + cfg['backbone_type'] + '_' + str(is_only_arc) + '_' + str(cfg['m']) + 'x' + str(
%        cfg['q'])+
%ijbc_1n('gallery_res50_incev2_ids','gallery_res50_incev2_feats','probe_res50_incev2_ids','probe_res50_incev2_feats')
%ijbc_1n('gallery_res50_xception_ids','gallery_res50_xception_feats','probe_res50_xception_ids','probe_res50_xception_feats')
%ijbc_1n('gallery_incev2_xception_ids','gallery_incev2_xception_feats','probe_incev2_xception_ids','probe_incev2_xception_feats')

load('../data/'+gallery_feats_path+'.csv')
load('../data/'+gallery_ids_path+'.csv')
load('../data/'+probe_feats_path+'.csv')
load('../data/'+probe_ids_path+'.csv')



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


final_dist =pdist2( gallery_feats,probe_feats);

[iom_DIR(:,:,2), iom_osiFAR(2,:)] = OpenSetROC(1-final_dist , gallery_ids, probe_ids, osiFarPoints );


iom_DIR(1,[11 20],2) * 100



end




%addpath('../');
%addpath('matlab_tools')
%addpath_recurse('BLUFR')
%addpath_recurse('btp')
%addpath('k_reciprocal_re_ranking')
%
%%%
%k1 = 20;
%k2 = 6;
%lambda = 0.3;
%measure = 'Hamming';
%% ds = "LFW";
%log_path = "logs/log_ijbc_idnew.log";
%
%%%
%Descriptor_G1 = importdata("../embeddings_dl/Insight_ResNet100_ijbc_G1_feat_dlIoM_512x8.csv");
%Descriptor_G2 = importdata("../embeddings_dl/Insight_ResNet100_ijbc_G2_feat_dlIoM_512x8.csv");
%Descriptor_P = importdata("../embeddings_dl/Insight_ResNet100_ijbc_P_feat_dlIoM_512x8.csv");
%
%Names_G1 = importdata("../embeddings_dl/Insight_ResNet100_ijbc_G1_name_dl_512x8.txt");
%Names_G2 = importdata("../embeddings_dl/Insight_ResNet100_ijbc_G2_name_dl_512x8.txt");
%Names_P = importdata("../embeddings_dl/Insight_ResNet100_ijbc_P_name_dl_512x8.txt");
%
%
%final_dist_o2 =(pdist2( Descriptor_G1,Descriptor_P,  measure));
%
%% final_dist_o2_re = re_ranking_score(final_dist_o2',Names_G,Names_P,Descriptor_G,Descriptor_P, k1, k2, lambda,measure);
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
%%% Compute the cosine similarity score between the test samples.
%
%[iom_DIR(:,:,2), iom_osiFAR(2,:)] = OpenSetROC(1-final_dist_o2, Names_G1, Names_P, osiFarPoints );
%[iom_max_rank,iom_rec_rates] = CMC(1-final_dist_o2',Names_P,Names_G1);
%
%% [iom_DIR_re(:,:,2), iom_osiFAR(2,:)] = OpenSetROC(1-final_dist_o2_re, Names_G, Names_P, osiFarPoints );
%
%perf_id = [ iom_DIR(1,[11 20],2) * 100 ];
%
%%%
%Descriptor_G = [Descriptor_G1;Descriptor_G2];
%Names_G = [Names_G1;Names_G2];
%
%
%final_dist =(pdist2( Descriptor_G,Descriptor_P,  measure));
%
%% [CMC_eu_re, map_eu_re, ~, ~] = evaluation(final_dist_re, facenet_gallery_label, facenet_probe_label_c, [], []);
%% [CMC_eu, map_eu, ~, ~] = evaluation(final_dist', facenet_gallery_label, facenet_probe_label_c, [], []);
%[iom_max_rank,iom_rec_rates] = CMC(1-final_dist',Names_P,Names_G);
%score_avg_mAP_iom = []; % open-set identification false accept rates of the 10 trials
%score_avg_mAP_iom_re = []; % open-set identification false accept rates of the 10 trials
%for k2=[1:10]
%    score_avg_mAP_iom = [score_avg_mAP_iom average_precision(final_dist',Names_G==Names_P',k2)];
%end
%
%[iom_VR(1,:), iom_veriFAR(1,:)] = EvalROC(1-final_dist, Names_G, Names_P, veriFarPoints);
%
%perf_vr = [iom_rec_rates(1)* 100 iom_VR(1,[29 38 56])* 100 iom_DIR(1,[11 20],2) * 100 score_avg_mAP_iom(1:5) ];
%
