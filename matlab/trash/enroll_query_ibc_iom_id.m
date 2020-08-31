function enroll_query_ijbc_iom_id(hashcode_path,filename_path)
% hashcode_path e.g. res50_lfw_feat_dIoM_512x2.csv
% filename_path e.g. lresnet100e_ir_lfw_name.txt
% e.g. enroll_query_iom lresnet100e_ir_lfw_feat_dIoM_512x2.csv  lresnet100e_ir_lfw_name.txt
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
%%

Descriptor_orig = importdata("../embeddings/"+hashcode_path);
fid_lfw_name=importdata("../embeddings/" + filename_path);

[hash_facenet_probe_c,hash_facenet_probe_o1,hash_facenet_probe_o2,hash_facenet_probe_o3,hash_facenet_gallery,facenet_probe_label_c,facenet_probe_label_o1,facenet_probe_label_o2,facenet_probe_label_o3, facenet_gallery_label] = generateDataset(ds,Descriptor_orig,fid_lfw_name)



%%%% generate identifier, dimension same to hash code
[identifiers ] = generate_identifier2(m,q,6000);
%%% mixing gallery
mixing_facenet_gallery = [];
for i = progress(1:size(facenet_gallery_label,2))
    mixing_facenet_gallery(i,:) = bitxor(hash_facenet_gallery(i,:),identifiers(facenet_gallery_label(i),:));
end

%%

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

final_dist = zeros(size(facenet_probe_label_c,2),size(mixing_facenet_gallery,1));
for i = progress(1:size(facenet_probe_label_c,2))
    dist = zeros(1,size(mixing_facenet_gallery,1));
    for j=1: size(mixing_facenet_gallery,1)
        gallery_bin =  mixing_facenet_gallery(j,:);
        retrieved_id = bitxor(gallery_bin,hash_facenet_probe_c(i,:));
        dist(j) = sum(bitxor(retrieved_id,identifiers(facenet_gallery_label(j),:)))/m;
    end
    final_dist(i,:) = dist;

end


final_dist_o1 = zeros(size(facenet_probe_label_o1,2),size(mixing_facenet_gallery,1));
for i = progress(1:size(facenet_probe_label_o1,2))
    dist =  zeros(1,size(mixing_facenet_gallery,1));
    for j=1: size(mixing_facenet_gallery,1)
        gallery_bin =  mixing_facenet_gallery(j,:);
        retrieved_id = bitxor(gallery_bin,hash_facenet_probe_o1(i,:));
        dist(j) = sum(bitxor(retrieved_id,identifiers(facenet_gallery_label(j),:)))/m;
    end
    final_dist_o1(i,:) = dist;
end


final_dist_o2 = zeros(size(facenet_probe_label_o2,2),size(mixing_facenet_gallery,1));
for i = progress(1:size(facenet_probe_label_o2,2))

    dist =  zeros(1,size(mixing_facenet_gallery,1));
    for j=1: size(mixing_facenet_gallery,1)
        gallery_bin =  mixing_facenet_gallery(j,:);
        retrieved_id = bitxor(gallery_bin,hash_facenet_probe_o2(i,:));
        dist(j) = sum(bitxor(retrieved_id,identifiers(facenet_gallery_label(j),:)))/m;
    end
    final_dist_o2(i,:) = dist;
end


final_dist_o3 = zeros(size(facenet_probe_label_o3,2),size(mixing_facenet_gallery,1));
for i = progress(1:size(facenet_probe_label_o3,2))
    dist =  zeros(1,size(mixing_facenet_gallery,1));
    for j=1: size(mixing_facenet_gallery,1)
        gallery_bin =  mixing_facenet_gallery(j,:);
        retrieved_id = bitxor(gallery_bin,hash_facenet_probe_o3(i,:));
        dist(j) = sum(bitxor(retrieved_id,identifiers(facenet_gallery_label(j),:)))/m;
    end
    final_dist_o3(i,:) = dist;
end

% Evaluate the open-set identification performance.
% Evaluate the verification performance.
% CMC close set

final_dist_re = re_ranking_score(final_dist,facenet_gallery_label,facenet_probe_label_c,mixing_facenet_gallery,hash_facenet_probe_c, k1, k2, lambda,measure);
final_dist_o1_re = re_ranking_score(final_dist_o1,facenet_gallery_label,facenet_probe_label_o1,mixing_facenet_gallery,hash_facenet_probe_o1, k1, k2, lambda,measure);
final_dist_o2_re = re_ranking_score(final_dist_o2,facenet_gallery_label,facenet_probe_label_o2,mixing_facenet_gallery,hash_facenet_probe_o2, k1, k2, lambda,measure);
final_dist_o3_re = re_ranking_score(final_dist_o3,facenet_gallery_label,facenet_probe_label_o3,mixing_facenet_gallery,hash_facenet_probe_o3, k1, k2, lambda,measure);

% [CMC_eu_re, map_eu_re, ~, ~] = evaluation(final_dist_re, facenet_gallery_label, facenet_probe_label_c, [], []);
% [CMC_eu, map_eu, ~, ~] = evaluation(final_dist', facenet_gallery_label, facenet_probe_label_c, [], []);
[iom_max_rank,iom_rec_rates] = CMC(1-final_dist,facenet_probe_label_c,facenet_gallery_label);
[iom_max_rank,iom_rec_rates_re] = CMC(1-final_dist_re',facenet_probe_label_c,facenet_gallery_label);

score_avg_mAP_iom = []; % open-set identification false accept rates of the 10 trials
score_avg_mAP_iom_re = []; % open-set identification false accept rates of the 10 trials
for k2=[1:10 20:10:100 200:100:1000]
    score_avg_mAP_iom = [score_avg_mAP_iom average_precision(final_dist,facenet_gallery_label==facenet_probe_label_c',k2)];
    score_avg_mAP_iom_re = [score_avg_mAP_iom_re average_precision(final_dist_re',facenet_gallery_label==facenet_probe_label_c',k2)];
end

[iom_VR(1,:), iom_veriFAR(1,:)] = EvalROC(1-final_dist', facenet_gallery_label, facenet_probe_label_c, veriFarPoints);
[iom_DIR(:,:,1), iom_osiFAR(1,:)] = OpenSetROC(1-final_dist_o1', facenet_gallery_label, facenet_probe_label_o1, osiFarPoints );
[iom_DIR(:,:,2), iom_osiFAR(2,:)] = OpenSetROC(1-final_dist_o2', facenet_gallery_label, facenet_probe_label_o2, osiFarPoints );
[iom_DIR(:,:,3), iom_osiFAR(3,:)] = OpenSetROC(1-final_dist_o3', facenet_gallery_label, facenet_probe_label_o3, osiFarPoints );

[iom_VR_re(1,:), iom_veriFAR(1,:)] = EvalROC(1-final_dist_re, facenet_gallery_label, facenet_probe_label_c, veriFarPoints);
[iom_DIR_re(:,:,1), iom_osiFAR(1,:)] = OpenSetROC(1-final_dist_o1_re, facenet_gallery_label, facenet_probe_label_o1, osiFarPoints );
[iom_DIR_re(:,:,2), iom_osiFAR(2,:)] = OpenSetROC(1-final_dist_o2_re, facenet_gallery_label, facenet_probe_label_o2, osiFarPoints );
[iom_DIR_re(:,:,3), iom_osiFAR(3,:)] = OpenSetROC(1-final_dist_o3_re, facenet_gallery_label, facenet_probe_label_o3, osiFarPoints );


perf = [reportVR reportDIR iom_rec_rates(1)* 100 iom_VR(1,[29 38 56])* 100 iom_DIR(1,[11 20],1) * 100 iom_DIR(1,[11 20],2) * 100 iom_DIR(1,[11 20],3) * 100 score_avg_mAP_iom(1:5) iom_rec_rates_re(1)* 100 iom_VR_re(1,[29 38 56])* 100 iom_DIR_re(1,[11 20],1) * 100 iom_DIR_re(1,[11 20],2) * 100 iom_DIR_re(1,[11 20],3) * 100 score_avg_mAP_iom_re(1:5)];
fid=fopen('logs/log_lfw_iom_id.txt','a');
fwrite(fid,hashcode_path+" ");
fclose(fid)
dlmwrite('logs/log_lfw_iom_id.txt', perf, '-append');
end