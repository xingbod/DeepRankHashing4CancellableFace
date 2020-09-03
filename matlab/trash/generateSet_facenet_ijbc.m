load('../facedataset/ijbc_256_facenet_embeddings.mat')
load('../facedataset/ijbclables.mat')

fusion_embedding=ijbc_256_facenet_embeddings;
Zfusion_embedding = zscore(fusion_embedding,0,2);
Descriptors1 = Zfusion_embedding/norm(Zfusion_embedding);

M = containers.Map({'abc'},{[]});
for i=1:length(ijbclables)
    if isKey(M,char(ijbclables(i)))
        M(char(ijbclables(i))) = [M(char(ijbclables(i))); Descriptors1(i,:)];
    else
        M(char(ijbclables(i)))=Descriptors1(i,:);
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
    if cnt>=30
        known(allnames{nameidx})=  M(allnames{nameidx});
    elseif cnt>15
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

save('data/facenet_train_set.mat','facenet_train_set');
save('data/facenet_train_label.mat','facenet_train_label');
save('data/facenet_gallery.mat','facenet_gallery');
save('data/facenet_gallery_label.mat','facenet_gallery_label');
save('data/facenet_probe_c.mat','facenet_probe_c');
save('data/facenet_probe_label_c.mat','facenet_probe_label_c');
save('data/facenet_probe_o1.mat','facenet_probe_o1');
save('data/facenet_probe_o2.mat','facenet_probe_o2');
save('data/facenet_probe_o3.mat','facenet_probe_o3');
save('data/facenet_probe_label_o1.mat','facenet_probe_label_o1');
save('data/facenet_robe_label_o2.mat','facenet_probe_label_o2');
save('data/facenet_probe_label_o3.mat','facenet_probe_label_o3');
