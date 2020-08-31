function [Descriptors,lfwlables] = generate_lfw_align(Descriptor_orig,fid_lfw_name)
%GENERATE_LFW_ALIGN Summary of this function goes here
%   Detailed explanation goes here
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

load('data/lfw_label.mat')%lfwlables

Descriptors = align_lfw_feat_dIoM;

end

