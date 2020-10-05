DS ="VGG2"
backbone = "InceptionResNetV2"
if DS == "IJBC"
    embed = "ijbc";
elseif DS == "VGG2"
    embed = "VGG2";
else DS == "LFW"
    embed = "lfw";
end
ds = DS
measure = "Euclidean";
remark = "deep_orig_feat";

feat_path = "embeddings_0831/"+backbone+"_"+embed+"_feat.csv"

filename_path = 'embeddings_0831/'+backbone+'_'+embed+'_name.txt'

addpath('../');
addpath('matlab_tools')
addpath_recurse('BLUFR')
addpath_recurse('btp')
addpath('k_reciprocal_re_ranking')

%%
k1 = 20;
k2 = 6;
lambda = 0.3;
% measure = 'Euclidean';
% ds = "LFW";
% remark = "deepfeat";
log_path = "logs/log_"+ds+"_"+remark+".log";
%%

Descriptor_orig = importdata("../"+feat_path);
fid_lfw_name=importdata("../" + filename_path);



mydist =pdist2(Descriptor_orig(1:2:51786,:),Descriptor_orig(1:2:51786,:));
myname = string(cell2mat(fid_lfw_name(1:2:51786)));
mysim = 1- mydist/80;

imposter = mysim(myname~=myname');
gen = mysim(myname==myname');
[EER, mTSR, mFAR, mFRR, mGAR,EERthreshold] = computeperformance(gen, imposter(randperm(670369,70000)), 0.01);

samples_per_user = 6;
known = Descriptor_orig(1:2000*samples_per_user,:);
known_unknowns = Descriptor_orig(2000*samples_per_user+1:4000*samples_per_user,:);
unknown_unknowns =Descriptor_orig(4000*samples_per_user+1:6000*samples_per_user,:);


%% train set and  gallery probe set
facenet_gallery=[known(1:samples_per_user:2000*samples_per_user,:);known(2:samples_per_user:2000*samples_per_user,:);known(3:samples_per_user:2000*samples_per_user,:)];
facenet_gallery_label=[1:2000 1:2000 1:2000];

S=[known(4:samples_per_user:2000*samples_per_user,:);known(5:samples_per_user:2000*samples_per_user,:);known(4:samples_per_user:2000*samples_per_user,:)];
S_label=[1:2000 1:2000 1:2000];

K=[ known_unknowns(2:samples_per_user:2000*samples_per_user,:); known_unknowns(3:samples_per_user:2000*samples_per_user,:); known_unknowns(4:samples_per_user:2000*samples_per_user,:)];
K_label=[2001:4000 2001:4000 2001:4000];

U=[unknown_unknowns(1:samples_per_user:2000*samples_per_user,:)];
U_label=[4001:6000];


facenet_probe_c=S;
facenet_probe_label_c=S_label;

facenet_probe_o1=[S ; K];
facenet_probe_label_o1=[S_label K_label];

facenet_probe_o2=[S;U];
facenet_probe_label_o2=[S_label U_label];

facenet_probe_o3=[S;K;U];
facenet_probe_label_o3=[S_label K_label U_label];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hash_facenet_probe_c=facenet_probe_c;
hash_facenet_probe_o1=[];
hash_facenet_probe_o2=facenet_probe_o2;
hash_facenet_probe_o3=[];
hash_facenet_gallery=facenet_gallery;



mydist =pdist2(hash_facenet_gallery,hash_facenet_probe_c);
mysim = 1- mydist/80;

imposter = mysim(facenet_probe_label_c~=facenet_gallery_label');
gen = mysim(facenet_probe_label_c==facenet_gallery_label');
[EER, mTSR, mFAR, mFRR, mGAR,EERthreshold] = computeperformance(gen, imposter(randperm(670369,70000)), 0.01);

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


