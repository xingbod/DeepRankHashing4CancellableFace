function [perf] = evaluate_searching(hash_facenet_probe_c,hash_facenet_probe_o1,hash_facenet_probe_o2,hash_facenet_probe_o3,hash_facenet_gallery,facenet_probe_label_c,facenet_probe_label_o1,facenet_probe_label_o2,facenet_probe_label_o3, facenet_gallery_label,measure)
%EVALUATE_SEARCHING_MIXDEMIX Summary of this function goes here
%   Detailed explanation goes here
%%%% generate identifier, dimension same to hash code
m = size(hash_facenet_probe_c,2);
q=max(max(hash_facenet_probe_c))+1;
k1 = 20;
k2 = 6;
lambda = 0.3;

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
final_dist =(pdist2( hash_facenet_gallery,hash_facenet_probe_c,  measure));
% final_dist_o1 =(pdist2( hash_facenet_gallery,hash_facenet_probe_o1,  measure));
final_dist_o2 =(pdist2( hash_facenet_gallery,hash_facenet_probe_o2,  measure));
% final_dist_o3 =(pdist2( hash_facenet_gallery,hash_facenet_probe_o3,  measure));


% Evaluate the open-set identification performance.
% Evaluate the verification performance.
% CMC close set

final_dist_re = re_ranking_score(final_dist',facenet_gallery_label,facenet_probe_label_c,hash_facenet_gallery,hash_facenet_probe_c, k1, k2, lambda,measure);
% final_dist_o1_re = re_ranking_score(final_dist_o1',facenet_gallery_label,facenet_probe_label_o1,hash_facenet_gallery,hash_facenet_probe_o1, k1, k2, lambda,measure);
final_dist_o2_re = re_ranking_score(final_dist_o2',facenet_gallery_label,facenet_probe_label_o2,hash_facenet_gallery,hash_facenet_probe_o2, k1, k2, lambda,measure);
% final_dist_o3_re = re_ranking_score(final_dist_o3',facenet_gallery_label,facenet_probe_label_o3,hash_facenet_gallery,hash_facenet_probe_o3, k1, k2, lambda,measure);

% [CMC_eu_re, map_eu_re, ~, ~] = evaluation(final_dist_re, facenet_gallery_label, facenet_probe_label_c, [], []);
% [CMC_eu, map_eu, ~, ~] = evaluation(final_dist', facenet_gallery_label, facenet_probe_label_c, [], []);
[iom_max_rank,iom_rec_rates] = CMC(1-final_dist',facenet_probe_label_c,facenet_gallery_label);
[iom_max_rank,iom_rec_rates_re] = CMC(1-final_dist_re',facenet_probe_label_c,facenet_gallery_label);

score_avg_mAP_iom = []; % open-set identification false accept rates of the 10 trials
score_avg_mAP_iom_re = []; % open-set identification false accept rates of the 10 trials
for k2=[1:10 20:10:100 200:100:1000]
    score_avg_mAP_iom = [score_avg_mAP_iom average_precision(final_dist',facenet_gallery_label==facenet_probe_label_c',k2)];
    score_avg_mAP_iom_re = [score_avg_mAP_iom_re average_precision(final_dist_re',facenet_gallery_label==facenet_probe_label_c',k2)];
end

[iom_VR(1,:), iom_veriFAR(1,:)] = EvalROC(1-final_dist, facenet_gallery_label, facenet_probe_label_c, veriFarPoints);
% [iom_DIR(:,:,1), iom_osiFAR(1,:)] = OpenSetROC(1-final_dist_o1, facenet_gallery_label, facenet_probe_label_o1, osiFarPoints );
[iom_DIR(:,:,2), iom_osiFAR(2,:)] = OpenSetROC(1-final_dist_o2, facenet_gallery_label, facenet_probe_label_o2, osiFarPoints );
% [iom_DIR(:,:,3), iom_osiFAR(3,:)] = OpenSetROC(1-final_dist_o3, facenet_gallery_label, facenet_probe_label_o3, osiFarPoints );

[iom_VR_re(1,:), iom_veriFAR(1,:)] = EvalROC(1-final_dist_re, facenet_gallery_label, facenet_probe_label_c, veriFarPoints);
% [iom_DIR_re(:,:,1), iom_osiFAR(1,:)] = OpenSetROC(1-final_dist_o1_re, facenet_gallery_label, facenet_probe_label_o1, osiFarPoints );
[iom_DIR_re(:,:,2), iom_osiFAR(2,:)] = OpenSetROC(1-final_dist_o2_re, facenet_gallery_label, facenet_probe_label_o2, osiFarPoints );
% [iom_DIR_re(:,:,3), iom_osiFAR(3,:)] = OpenSetROC(1-final_dist_o3_re, facenet_gallery_label, facenet_probe_label_o3, osiFarPoints );


perf = [iom_rec_rates(1)* 100 iom_VR(1,[29 38 56])* 100 iom_DIR(1,[11 20],2) * 100 score_avg_mAP_iom(1:5) iom_rec_rates_re(1)* 100 iom_VR_re(1,[29 38 56])* 100 iom_DIR_re(1,[11 20],2) * 100 score_avg_mAP_iom_re(1:5)];

end

