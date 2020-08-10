load('E:\my research\etri2020\arcface-tf2\matlab\data\lfw_feat.csv')
load('E:\my research\etri2020\arcface-tf2\matlab\data\lfw_name.csv')
load('E:\my research\etri2020\arcface-tf2\matlab\data\lfw_label.mat')

fid=importdata('E:\my research\etri2020\arcface-tf2\matlab\data\lfw_name.csv');
lfw_name=[];
for i=1:size(fid,1)
    lfw_name = [lfw_name; string(cell2mat(fid(i)))+".jpg"];
end

fid2=importdata('E:\my research\etri2020\arcface-tf2\matlab\BLUFR\list\lfw\image_list.txt');
lfw_name2=[];
for i=1:size(fid,1)
    lfw_name2 = [lfw_name2; string(cell2mat(fid2(i)))];
end

my_index=[];
for i=1:size(fid,1)
  indx = find(lfw_name(i)==lfw_name2);
    my_index = [my_index, indx];
end

align_lfw_feat = lfw_feat(my_index,:);
align_lfw_name = lfw_name(my_index);

save('align_lfw_feat.mat','align_lfw_feat')