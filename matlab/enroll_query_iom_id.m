function enroll_query_iom_id(hashcode_path,filename_path)
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
lfw_name=[];
for i=1:size(fid_lfw_name,1)
    lfw_name = [lfw_name; string(cell2mat(fid_lfw_name(i)))+".jpg"];
end

fid2=importdata('BLUFR/list/lfw/image_list.txt');
lfw_name2=[];
for i=1:size(fid_lfw_name,1)
    lfw_name2 = [lfw_name2; string(cell2mat(fid2(i)))];
end

my_index=[];
for i=1:size(fid_lfw_name,1)
    indx = find(lfw_name2(i)==lfw_name);
    my_index = [my_index, indx];
end

% align_lfw_feat = lfw_feat(my_index,:);
align_lfw_feat_dIoM = Descriptor_orig(my_index,:);
align_lfw_name = lfw_name(my_index);

% save('data/align_lfw_feat.mat','align_lfw_feat')
% save('data/align_lfw_feat_dIoM.mat','align_lfw_feat_dIoM')

% close all; clear; clc;
load('data/lfw_label.mat')
% load('data/align_lfw_feat_dIoM_512x2.mat')

Descriptors = align_lfw_feat_dIoM;

%% BLUFR
% [reportVeriFar, reportVR,reportRank, reportOsiFar, reportDIR] = LFW_BLUFR(Descriptors,'measure','Hamming');
reportVR = 0;
reportDIR = 0;
%% Voting protocol based on mixing
m = size(Descriptors,2);
q=max(max(Descriptors))+1;

M = containers.Map({'abc'},{[]});
for i=1:length(lfwlables)
    if isKey(M,char(lfwlables(i)))
        M(char(lfwlables(i))) = [M(char(lfwlables(i))); Descriptors(i,:)];
    else
        M(char(lfwlables(i)))=Descriptors(i,:);
    end
end
remove(M,'abc');

%% three group
allnames=M.keys;
known= containers.Map({'abc'},{[]});
known_unknowns= containers.Map({'abc'},{[]});
unknown_unknowns= containers.Map({'abc'},{[]});
for nameidx=1:length(allnames)
    thisuseremplate=M(allnames{nameidx});
    cnt=size(thisuseremplate,1);
    if cnt>=4
        known(allnames{nameidx})=  M(allnames{nameidx});
    elseif cnt>1
        known_unknowns(allnames{nameidx})=  M(allnames{nameidx});
    else
        unknown_unknowns(allnames{nameidx})=  M(allnames{nameidx});
    end
end
remove(known,'abc');
remove(known_unknowns,'abc');
remove(unknown_unknowns,'abc');

%% train set and  facenet_gallery probe set
facenet_train_set=[];
facenet_train_label=[];

facenet_gallery=[];
facenet_gallery_label=[];

known_names=known.keys;
for nameidx=1:length(known_names)
    thisuseremplate=known(known_names{nameidx});
    facenet_train_set = [facenet_train_set ;thisuseremplate(1:3,:) ];
    facenet_train_label=[facenet_train_label repmat(string(known_names{nameidx}),1,3)];
end

facenet_gallery = facenet_train_set;
facenet_gallery_label = facenet_train_label;

known_unknowns_names=known_unknowns.keys;
for nameidx=1:length(known_unknowns_names)
    thisuseremplate=known_unknowns(known_unknowns_names{nameidx});
    facenet_train_set = [facenet_train_set ;thisuseremplate(1,:) ];
    facenet_train_label=[facenet_train_label string(known_unknowns_names{nameidx})];
end
% remaining as facenet_probe_c
S=[];
S_label=[];
for nameidx=1:length(known_names)
    thisuseremplate=known(known_names{nameidx});
    cnt=size(thisuseremplate,1);
    S = [S ;thisuseremplate(4:end,:) ];
    S_label=[S_label repmat(string(known_names{nameidx}),1,cnt-3)];
end
% S union K  o1

K=[];
K_label=[];
for nameidx=1:length(known_unknowns_names)
    thisuseremplate=known_unknowns(known_unknowns_names{nameidx});
    cnt=size(thisuseremplate,1);
    K = [K ;thisuseremplate(2:end,:) ];
    K_label=[K_label repmat(string(known_unknowns_names{nameidx}),1,cnt-1)];
end

% S union U  o2
U=[];
U_label=[];
unknown_unknowns_names=unknown_unknowns.keys;
for nameidx=1:length(unknown_unknowns_names)
    thisuseremplate=unknown_unknowns(unknown_unknowns_names{nameidx});
    U = [U ;thisuseremplate(1,:) ];
    U_label=[U_label string(unknown_unknowns_names{nameidx})];
end

facenet_probe_c=S;
facenet_probe_label_c=S_label;

facenet_probe_o1=[S ; K];
facenet_probe_label_o1=[S_label K_label];

facenet_probe_o2=[S;U];
facenet_probe_label_o2=[S_label U_label];

facenet_probe_o3=[S;K;U];
facenet_probe_label_o3=[S_label K_label U_label];

%label trans to number
for nameidx=1:length(allnames)
    facenet_probe_label_c(find(facenet_probe_label_c==string(allnames{nameidx})))=nameidx;
    facenet_probe_label_o1(find(facenet_probe_label_o1==string(allnames{nameidx})))=nameidx;
    facenet_probe_label_o2(find(facenet_probe_label_o2==string(allnames{nameidx})))=nameidx;
    facenet_probe_label_o3(find(facenet_probe_label_o3==string(allnames{nameidx})))=nameidx;
    facenet_gallery_label(find(facenet_gallery_label==string(allnames{nameidx})))=nameidx;
end
% I also dont want to do so

facenet_probe_label_c = double(facenet_probe_label_c);
facenet_probe_label_o1 = double(facenet_probe_label_o1);
facenet_probe_label_o2 = double(facenet_probe_label_o2);
facenet_probe_label_o3 = double(facenet_probe_label_o3);
facenet_gallery_label = double(facenet_gallery_label);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hash_facenet_probe_c=facenet_probe_c;
hash_facenet_probe_o1=facenet_probe_o1;
hash_facenet_probe_o2=facenet_probe_o2;
hash_facenet_probe_o3=facenet_probe_o3;
hash_facenet_gallery=facenet_gallery;
%%%% generate identifier, dimension same to hash code
[identifiers ] = generate_identifier2(m,q,6000);
%%% mixing gallery
mixing_facenet_gallery = [];
parfor i =1:size(facenet_gallery_label,2)
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
parfor i = 1:size(facenet_probe_label_c,2)
    dist = zeros(1,size(mixing_facenet_gallery,1));
    for j=1: size(mixing_facenet_gallery,1)
        gallery_bin =  mixing_facenet_gallery(j,:);
        retrieved_id = bitxor(gallery_bin,hash_facenet_probe_c(i,:));
        dist(j) = sum(bitxor(retrieved_id,identifiers(facenet_gallery_label(j),:)))/m;
    end
    final_dist(i,:) = dist;

end


final_dist_o1 = zeros(size(facenet_probe_label_o1,2),size(mixing_facenet_gallery,1));
parfor i = progress(1:size(facenet_probe_label_o1,2))
    dist =  zeros(1,size(mixing_facenet_gallery,1));
    for j=1: size(mixing_facenet_gallery,1)
        gallery_bin =  mixing_facenet_gallery(j,:);
        retrieved_id = bitxor(gallery_bin,hash_facenet_probe_o1(i,:));
        dist(j) = sum(bitxor(retrieved_id,identifiers(facenet_gallery_label(j),:)))/m;
    end
    final_dist_o1(i,:) = dist;
end


final_dist_o2 = zeros(size(facenet_probe_label_o2,2),size(mixing_facenet_gallery,1));
parfor i = 1:size(facenet_probe_label_o2,2)

    dist =  zeros(1,size(mixing_facenet_gallery,1));
    for j=1: size(mixing_facenet_gallery,1)
        gallery_bin =  mixing_facenet_gallery(j,:);
        retrieved_id = bitxor(gallery_bin,hash_facenet_probe_o2(i,:));
        dist(j) = sum(bitxor(retrieved_id,identifiers(facenet_gallery_label(j),:)))/m;
    end
    final_dist_o2(i,:) = dist;
end

tic
final_dist_o3 = zeros(size(facenet_probe_label_o3,2),size(mixing_facenet_gallery,1));
parfor i = 1:size(facenet_probe_label_o3,2)
    dist =  zeros(1,size(mixing_facenet_gallery,1));
    for j=1: size(mixing_facenet_gallery,1)
        gallery_bin =  mixing_facenet_gallery(j,:);
        retrieved_id = bitxor(gallery_bin,hash_facenet_probe_o3(i,:));
        dist(j) = sum(bitxor(retrieved_id,identifiers(facenet_gallery_label(j),:)))/m;
    end
    final_dist_o3(i,:) = dist;
end
toc
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

[iom_VR_re(1,:), iom_veriFAR(1,:)] = EvalROC(1-final_dist_re', facenet_gallery_label, facenet_probe_label_c, veriFarPoints);
[iom_DIR_re(:,:,1), iom_osiFAR(1,:)] = OpenSetROC(1-final_dist_o1_re', facenet_gallery_label, facenet_probe_label_o1, osiFarPoints );
[iom_DIR_re(:,:,2), iom_osiFAR(2,:)] = OpenSetROC(1-final_dist_o2_re', facenet_gallery_label, facenet_probe_label_o2, osiFarPoints );
[iom_DIR_re(:,:,3), iom_osiFAR(3,:)] = OpenSetROC(1-final_dist_o3_re', facenet_gallery_label, facenet_probe_label_o3, osiFarPoints );


perf = [reportVR reportDIR iom_rec_rates(1)* 100 iom_VR(1,[29 38 56])* 100 iom_DIR(1,[11 20],1) * 100 iom_DIR(1,[11 20],2) * 100 iom_DIR(1,[11 20],3) * 100 score_avg_mAP_iom(1:5)
    iom_rec_rates_re(1)* 100 iom_VR_re(1,[29 38 56])* 100 iom_DIR_re(1,[11 20],1) * 100 iom_DIR_re(1,[11 20],2) * 100 iom_DIR_re(1,[11 20],3) * 100 score_avg_mAP_iom_re(1:5)]
fid=fopen('logs/log_hashing_identification.txt','a');
fwrite(fid,hashcode_path+" ");
fclose(fid)
dlmwrite('logs/log_hashing_identification.txt', perf, '-append');
end