clear all;

addpath('../');
addpath('matlab_tools')
addpath_recurse('BLUFR')

combs = nchoosek(0:19,2);
for cnt_i=1:50
       cnt_i
    fid_lfw_name = importdata('../embeddings_0831/Xception_lfw_name_512x8.txt');
    Descriptor_orig = importdata('../embeddings_0831/InceptionResNetV2_lfw_feat_LUT_PERM_512x8_'+string(combs(cnt_i,1))+'.csv');
    Descriptor_orig2 = importdata('../embeddings_0831/InceptionResNetV2_lfw_feat_LUT_PERM_512x8_'+string(combs(cnt_i,2))+'.csv');
    
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
    Descriptors = Descriptor_orig(my_index,:);
    Descriptors2 = Descriptor_orig2(my_index,:);
    align_lfw_name = lfw_name(my_index);
    
    % close all; clear; clc;
    load('data/lfw_label.mat')
    % load('data/align_lfw_feat_dIoM_512x2.mat')
    
    dist_all = 1-pdist2(Descriptors,Descriptors2,'Hamming');
    dist_gen = 1-pdist2(Descriptors,Descriptors,'Hamming');
    
    mask = lfwlables==lfwlables';
%
%    psedo_mated = dist_all(mask);
%    psedo_non_mated = dist_all(mask==0);
%    psedo_non_mated = psedo_non_mated(randperm(174614542,length(psedo_mated)));
%
    genuine = dist_gen(mask);
    imposter_mated = dist_all(mask);
    imposter = dist_gen(mask==0);
    imposter = imposter(randperm(174614542,length(genuine)));

    dlmwrite('permlut_genuine.txt',genuine,'-append')
    dlmwrite('permlut_imposter_mated.txt',imposter_mated,'-append')
    dlmwrite('permlut_imposter.txt',imposter,'-append')
%     dlmwrite('psedo_mated.txt',psedo_mated,'-append')
%     dlmwrite('psedo_non_mated.txt',psedo_non_mated,'-append')
    
end
%
% [EER, mTSR, mFAR, mFRR, mGAR] = computeperformance(psedo_mated, psedo_non_mated, 0.001);
% plothisf(psedo_mated,psedo_non_mated,'bit',1,1,10000);%EER 0.0.49 %
