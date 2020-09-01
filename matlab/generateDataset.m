function [hash_facenet_probe_c,hash_facenet_probe_o1,hash_facenet_probe_o2,hash_facenet_probe_o3,hash_facenet_gallery,facenet_probe_label_c,facenet_probe_label_o1,facenet_probe_label_o2,facenet_probe_label_o3, facenet_gallery_label] = generateDataset(ds,Descriptor_orig,fid_lfw_name)


if ds=="LFW"
    
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
    
    
    hash_facenet_probe_c=facenet_probe_c;
    hash_facenet_probe_o1=facenet_probe_o1;
    hash_facenet_probe_o2=facenet_probe_o2;
    hash_facenet_probe_o3=facenet_probe_o3;
    hash_facenet_gallery=facenet_gallery;
    
    
elseif ds == "VGG2"
    
    known = Descriptor_orig(1:2000*50,:);
    known_unknowns = Descriptor_orig(2000*50+1:4000*50,:);
    unknown_unknowns =Descriptor_orig(4000*50+1:6000*50,:);
    
    %% train set and  gallery probe set
    facenet_train_set = [known(1:50:2000*50,:); known(2:50:2000*50,:); known_unknowns(1:50:2000*50,:)];
    facenet_train_label=[1:2000 1:2000 2001:4000];
    
    facenet_gallery=[known(1:50:2000*50,:);known(2:50:2000*50,:);known(3:50:2000*50,:)];
    facenet_gallery_label=[1:2000 1:2000 1:2000];
    
    S=[known(4:50:2000*50,:);known(5:50:2000*50,:);known(4:50:2000*50,:)];
    S_label=[1:2000 1:2000 1:2000];
    
    K=[ known_unknowns(2:50:2000*50,:); known_unknowns(3:50:2000*50,:); known_unknowns(4:50:2000*50,:)];
    K_label=[2001:4000 2001:4000 2001:4000];
    
    U=[unknown_unknowns(1:50:2000*50,:)];
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
    hash_facenet_probe_o1=facenet_probe_o1;
    hash_facenet_probe_o2=facenet_probe_o2;
    hash_facenet_probe_o3=facenet_probe_o3;
    hash_facenet_gallery=facenet_gallery;
    
elseif ds == "IJBC"
    ijbclables = fid_lfw_name;
    M = containers.Map("0",{[]});
    for i=1:length(ijbclables)
        if isKey(M,string(ijbclables(i)))
            M(string(ijbclables(i))) = [M(string(ijbclables(i))); Descriptor_orig(i,:)];
        else
            M(string(ijbclables(i)))=Descriptor_orig(i,:);
        end
    end
    remove(M,"0");
    
    %% three group
    allnames=M.keys;
    known= containers.Map("abc",{[]});
    known_unknowns= containers.Map("abc",{[]});
    unknown_unknowns= containers.Map("abc",{[]});
    for nameidx=1:length(allnames)
        thisuseremplate=M(allnames{nameidx});
        cnt=size(thisuseremplate,1);
        if cnt>=10
            known(allnames{nameidx})=  M(allnames{nameidx});
        elseif cnt>5
            known_unknowns(allnames{nameidx})=  M(allnames{nameidx});
        else
            unknown_unknowns(allnames{nameidx})=  M(allnames{nameidx});
        end
    end
    remove(known,"abc");
    remove(known_unknowns,"abc");
    remove(unknown_unknowns,"abc");
    
    %% train set and  facenet_gallery probe set
    facenet_train_set=[];
    facenet_train_label=[];
    
    facenet_gallery=[];
    facenet_gallery_label=[];
    
    known_names=known.keys;
    for nameidx=1:length(known_names)
        thisuseremplate=known(known_names{nameidx});
        facenet_train_set = [facenet_train_set ;thisuseremplate(1:2,:) ];
        facenet_train_label=[facenet_train_label repmat(string(known_names{nameidx}),1,2)];
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
        S = [S ;thisuseremplate(4:6,:) ];
        S_label=[S_label repmat(string(known_names{nameidx}),1,3)];
    end
    % S union K  o1
    
    K=[];
    K_label=[];
    for nameidx=1:length(known_unknowns_names)
        thisuseremplate=known_unknowns(known_unknowns_names{nameidx});
        cnt=size(thisuseremplate,1);
        K = [K ;thisuseremplate(2:4,:) ];
        K_label=[K_label repmat(string(known_unknowns_names{nameidx}),1,3)];
    end
    
    % S union U  o2
    U=[];
    U_label=[];
    unknown_unknowns_names=unknown_unknowns.keys;
    for nameidx=1:length(unknown_unknowns_names)
        thisuseremplate=unknown_unknowns(unknown_unknowns_names{nameidx});
        cnt=size(thisuseremplate,1);
        if(cnt>2)
            cnt=3;
        end
        U = [U ;thisuseremplate(1:cnt,:) ];
        U_label=[U_label repmat(string(unknown_unknowns_names{nameidx}),1,cnt)];
        
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
end

end