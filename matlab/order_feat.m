addpath('../')
load('../embeddings/res50_lfw_feat_dIoM_512x2.csv')
% load('E:\my research\etri2020\arcface-tf2\matlab\data\lfw_name.csv')
% load('E:\my research\etri2020\arcface-tf2\matlab\data\lfw_label.mat')

fid=importdata('../embeddings/lfw_name.txt');
lfw_name=[];
for i=1:size(fid,1)
    lfw_name = [lfw_name; string(cell2mat(fid(i)))+".jpg"];
end

fid2=importdata('BLUFR/list/lfw/image_list.txt');
lfw_name2=[];
for i=1:size(fid,1)
    lfw_name2 = [lfw_name2; string(cell2mat(fid2(i)))];
end

my_index=[];
for i=1:size(fid,1)
  indx = find(lfw_name2(i)==lfw_name);
    my_index = [my_index, indx];
end

% align_lfw_feat = lfw_feat(my_index,:);
align_lfw_feat_dIoM_512x2 = lfw_feat_dIoM_512x2(my_index,:);
align_lfw_name = lfw_name(my_index);

% save('data/align_lfw_feat.mat','align_lfw_feat')
save('data/align_lfw_feat_dIoM_512x2.mat','align_lfw_feat_dIoM_512x2')