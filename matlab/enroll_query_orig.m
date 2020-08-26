function enroll_query_orig(feat_path,filename_path)
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
measure = 'Euclidean';
%%

Descriptor_orig = importdata("../embeddings/"+feat_path);
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
[reportVeriFar, reportVR,reportRank, reportOsiFar, reportDIR] = LFW_BLUFR(Descriptors,'measure',measure);

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


%
% facenet_probe_c
% facenet_probe_label_c
%
% facenet_probe_o1
% facenet_probe_label_o1
%
% facenet_probe_o2
% facenet_probe_label_o2
%
% facenet_probe_o3
% facenet_probe_label_o3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hash_facenet_probe_c=facenet_probe_c;
hash_facenet_probe_o1=facenet_probe_o1;
hash_facenet_probe_o2=facenet_probe_o2;
hash_facenet_probe_o3=facenet_probe_o3;
hash_facenet_gallery=facenet_gallery;
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
iom_veriFAR = zeros(numTrials, numVeriFarPoints); % verification false accept rates of the 10 trials

numOsiFarPoints = length(osiFarPoints);
numRanks = length(rankPoints);
iom_DIR = zeros(numRanks, numOsiFarPoints, numTrials); % detection and identification rates of the 10 trials
iom_osiFAR = zeros(numTrials, numOsiFarPoints); % open-set identification false accept rates of the 10 trials

% CMC close set
probFea = hash_facenet_probe_c';
galFea = hash_facenet_gallery';
cam_gallery = [];
cam_query = [];
label_gallery = facenet_gallery_label;
label_query = facenet_probe_label_c;
%% Euclidean

% Compute the cosine similarity score between the test samples.
facenet_score_c =1-(pdist2( hash_facenet_gallery,hash_facenet_probe_c,  measure))/100;
facenet_score_o1 =1-(pdist2( hash_facenet_gallery,hash_facenet_probe_o1,  measure))/100;
facenet_score_o2 =1-(pdist2( hash_facenet_gallery,hash_facenet_probe_o2,  measure))/100;
facenet_score_o3 =1-(pdist2( hash_facenet_gallery,hash_facenet_probe_o3,  measure))/100;


%dist_eu = pdist2(galFea', probFea');
% my_pdist2 = @(A, B) sqrt( bsxfun(@plus, sum(A.^2, 2), sum(B.^2, 2)') - 2*(A*B'));
% dist_eu = my_pdist2(galFea', probFea');
dist_eu = pdist2(galFea', probFea',measure);
[CMC_eu, map_eu, ~, ~] = evaluation(dist_eu, label_gallery, label_query, cam_gallery, cam_query);

fprintf(['The Euclidean performance:\n']);
fprintf(' Rank1,  mAP\n');
fprintf('%5.2f%%, %5.2f%%\n\n', CMC_eu(1) * 100, map_eu(1)*100);

%% Euclidean + re-ranking
query_num = size(probFea, 2);
dist_eu_re = re_ranking( [probFea galFea], 1, 1, query_num, k1, k2, lambda,measure);
[CMC_eu_re, map_eu_re, ~, ~] = evaluation(dist_eu_re, label_gallery, label_query, cam_gallery, cam_query);

fprintf(['The Euclidean + re-ranking performance:\n']);
fprintf(' Rank1,  mAP\n');
fprintf('%5.2f%%, %5.2f%%\n\n', CMC_eu_re(1) * 100, map_eu_re(1)*100);

% Evaluate the open-set identification performance.

% Evaluate the verification performance.
[iom_VR(1,:), iom_veriFAR(1,:)] = EvalROC(1-facenet_score_c, facenet_gallery_label, facenet_probe_label_c, veriFarPoints);

[iom_DIR(:,:,1), iom_osiFAR(1,:)] = OpenSetROC(1-facenet_score_o1, facenet_gallery_label, facenet_probe_label_o1, osiFarPoints );
[iom_DIR(:,:,2), iom_osiFAR(2,:)] = OpenSetROC(1-facenet_score_o2, facenet_gallery_label, facenet_probe_label_o2, osiFarPoints );
[iom_DIR(:,:,3), iom_osiFAR(3,:)] = OpenSetROC(1-facenet_score_o3, facenet_gallery_label, facenet_probe_label_o3, osiFarPoints );

% 
match_similarity =1-facenet_score_c';
[iom_max_rank,iom_rec_rates] = CMC(match_similarity,facenet_probe_label_c,facenet_gallery_label);
% 
% 
% score_avg_mAP_iom = []; % open-set identification false accept rates of the 10 trials
% for k2=[1:10 20:10:100 200:100:1000]
%     score_avg_mAP_iom = [score_avg_mAP_iom average_precision(final_dist,facenet_gallery_label==facenet_probe_label_c',k2)];
% end
% 
% 
% fprintf('avg_mAP_iom %8.5f\n', score_avg_mAP_iom(5)) % 注意输出格式前须有%符号，


% 
% save("data/"+hashcode_path+"_iom_veriFAR.mat","iom_veriFAR");
% save("data/"+hashcode_path+"_iom_max_rank.mat","iom_max_rank");
% save("data/"+hashcode_path+"_iom_VR.mat","iom_VR");
% save("data/"+hashcode_path+"_iom_rec_rates.mat","iom_rec_rates");
% save("data/"+hashcode_path+"_iom_osiFAR.mat","iom_osiFAR");
% save("data/"+hashcode_path+"_iom_DIR.mat","iom_DIR");

%% Display the benchmark performance and output to the log file.
% str = sprintf('Verification:\n');
% str = sprintf('%s\t@ FAR = %g%%: VR = %.2f%%.\n', str, reportVeriFar*100, reportVR);
% 
% str = sprintf('%sOpen-set Identification:\n', str);
% str = sprintf('%s\t@ Rank = %d, FAR = %g%%: DIR = %.2f%%.\n\n', str, reportRank, reportOsiFar*100, reportDIR);
% 

perf = [CMC_eu(1) * 100  map_eu(1)*100 CMC_eu_re(1) * 100 map_eu_re(1)*100 reportVR reportDIR iom_VR(1,[29 38 56])* 100 iom_rec_rates(1)* 100 iom_DIR(1,[11 20],1) * 100 iom_DIR(1,[11 20],2) * 100 iom_DIR(1,[11 20],3) * 100 ]%score_avg_mAP_iom(1:5)
fid=fopen('logs/log_orig.txt','a');
fwrite(fid,feat_path+" ");
fclose(fid)
dlmwrite('logs/log_orig.txt', perf, '-append');
end