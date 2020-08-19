function enroll_query_iom(hashcode_path,filename_path)
% hashcode_path e.g. res50_lfw_feat_dIoM_512x2.csv
% filename_path e.g. lresnet100e_ir_lfw_name.txt
% e.g. enroll_query_iom lresnet100e_ir_lfw_feat_dIoM_512x2.csv  lresnet100e_ir_lfw_name.txt
addpath('../');
addpath('matlab_tools')
addpath_recurse('BLUFR')
addpath_recurse('btp')

Descriptor = importdata("../embeddings/"+hashcode_path);
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
align_lfw_feat_dIoM = Descriptor(my_index,:);
align_lfw_name = lfw_name(my_index);

% save('data/align_lfw_feat.mat','align_lfw_feat')
% save('data/align_lfw_feat_dIoM.mat','align_lfw_feat_dIoM')

% close all; clear; clc;
load('data/lfw_label.mat')
% load('data/align_lfw_feat_dIoM_512x2.mat')

Descriptors = align_lfw_feat_dIoM;

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
    facenet_train_label(find(facenet_train_label==string(allnames{nameidx})))=nameidx;
    facenet_gallery_label(find(facenet_gallery_label==string(allnames{nameidx})))=nameidx;
end
% I also dont want to do so

facenet_probe_label_c = double(facenet_probe_label_c);
facenet_probe_label_o1 = double(facenet_probe_label_o1);
facenet_probe_label_o2 = double(facenet_probe_label_o2);
facenet_probe_label_o3 = double(facenet_probe_label_o3);
facenet_train_label = double(facenet_train_label);
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
%%%% generate identifier, dimension same to hash code
[identifiers ] = generate_identifier(m,q,6000);
%%% mixing gallery
mixing_facenet_gallery = [];
for i = progress(1:size(facenet_gallery_label,2))
    gallery_sample = dec2bin( hash_facenet_gallery(i,:),q)-'0';
    gallery_bin =reshape(gallery_sample',1,numel(gallery_sample));
    mixing_facenet_gallery(i,:) = bitxor(gallery_bin,identifiers(facenet_gallery_label(i),:));
end

%%
correct_ret=0;
incorrect_ret = 0;
final_dist = [];
for i = progress(1:size(facenet_probe_label_c,2))
    query_sample = dec2bin( hash_facenet_probe_c(i,:),q)-'0';
    query_bin =reshape(query_sample',1,numel(gallery_sample));
    
    dist = [];
    for j=1: size(mixing_facenet_gallery,1)
        gallery_bin =  mixing_facenet_gallery(j,:);
        retrieved_id = bitxor(gallery_bin,query_bin);
        dist = [dist pdist2(retrieved_id,identifiers(facenet_gallery_label(j),:),'Hamming')];
    end
    final_dist = [final_dist ; dist];
    [row column]=find(dist==min(dist(:)));
    if mode(facenet_gallery_label(column)) == facenet_probe_label_c(i)
        correct_ret = correct_ret+1;
    end
end

tar_c = correct_ret/size(facenet_probe_label_c,2);%  0.9517 96.90


score_avg_mAP_iom = []; % open-set identification false accept rates of the 10 trials
for k2=[1:10 20:10:100 200:100:1000]
    score_avg_mAP_iom = [score_avg_mAP_iom average_precision(final_dist,facenet_gallery_label==facenet_probe_label_c',k2)];
end


fprintf('avg_mAP_iom %8.5f\n', score_avg_mAP_iom(5)) % 注意输出格式前须有%符号，

correct_ret=0;
incorrect_ret = 0;

for i = progress(1:size(facenet_probe_label_o1,2))
    query_sample = dec2bin( hash_facenet_probe_o1(i,:),q)-'0';
    query_bin =reshape(query_sample',1,numel(gallery_sample));
    
    dist = [];
    for j=1: size(mixing_facenet_gallery,1)
        gallery_bin =  mixing_facenet_gallery(j,:);
        retrieved_id = bitxor(gallery_bin,query_bin);
        dist = [dist pdist2(retrieved_id,identifiers(facenet_gallery_label(j),:),'Hamming')];
    end
    [row column]=find(dist==min(dist(:)));
    if mode(facenet_gallery_label(column)) == facenet_probe_label_o1(i)
        correct_ret = correct_ret+1;
    end
end
tar_o1 = correct_ret/size(facenet_probe_label_o1,2);


correct_ret=0;
incorrect_ret = 0;

for i = progress(1:size(facenet_probe_label_o2,2))
    query_sample = dec2bin( hash_facenet_probe_o2(i,:),q)-'0';
    query_bin =reshape(query_sample',1,numel(gallery_sample));
    
    dist = [];
    for j=1: size(mixing_facenet_gallery,1)
        gallery_bin =  mixing_facenet_gallery(j,:);
        retrieved_id = bitxor(gallery_bin,query_bin);
        dist = [dist pdist2(retrieved_id,identifiers(facenet_gallery_label(j),:),'Hamming')];
    end
    [row column]=find(dist==min(dist(:)));
    if mode(facenet_gallery_label(column)) == facenet_probe_label_o2(i)
        correct_ret = correct_ret+1;
    end
end
tar_o2 = correct_ret/size(facenet_probe_label_o2,2);


correct_ret=0;
incorrect_ret = 0;

for i = progress(1:size(facenet_probe_label_o3,2))
    query_sample = dec2bin( hash_facenet_probe_o3(i,:),q)-'0';
    query_bin =reshape(query_sample',1,numel(gallery_sample));
    
    dist = [];
    for j=1: size(mixing_facenet_gallery,1)
        gallery_bin =  mixing_facenet_gallery(j,:);
        retrieved_id = bitxor(gallery_bin,query_bin);
        dist = [dist pdist2(retrieved_id,identifiers(facenet_gallery_label(j),:),'Hamming')];
    end
    [row column]=find(dist==min(dist(:)));
    if mode(facenet_gallery_label(column)) == facenet_probe_label_o3(i)
        correct_ret = correct_ret+1;
    end
end
tar_o3 = correct_ret/size(facenet_probe_label_o3,2);
fprintf('tar_c/mAP-c 1:5/tar_o1/tar_o2/tar_o3 %8.5f  %8.5f %8.5f %8.5f %8.5f\n', tar_c,score_avg_mAP_iom(1:5),tar_o1,tar_o2,tar_o3) % 注意输出格式前须有%符号，

end