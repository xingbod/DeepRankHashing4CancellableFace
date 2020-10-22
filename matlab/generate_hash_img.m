fid_lfw_name = importdata('../embeddings_0831/Xception_lfw_name_512x8.txt');
Descriptor_orig = importdata('../embeddings_0831/InceptionResNetV2_lfw_feat_LUT_512x8_0.csv');

 imshow(Descriptor_orig(242:283,1:100),[0,7])
 imshow(Descriptor_orig(32:50,1:100),[0,7])

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
    feature_used = zeros(length(allnames),512);
    for nameidx=1:length(allnames)
        nameidx
        thisuseremplate=M(allnames{nameidx});
        feature_used(nameidx,:) = thisuseremplate(1,:);
    end
 
    
imshow(feature_used(1:18,1:100),[0,7])