% load('facedataset\vgg2_512_insightface_embeddings.mat')
load('facedataset\vgg2_256_facenet_embeedings.mat')
% 
fusion_embedding=[vgg2_256_facenet_embeedings];
Zfusion_embedding = zscore(fusion_embedding,0,2);
Nfusion_embedding = Zfusion_embedding/norm(Zfusion_embedding);

known = Nfusion_embedding(1:2000*50,:);
known_unknowns = Nfusion_embedding(2000*50+1:4000*50,:);
unknown_unknowns =Nfusion_embedding(4000*50+1:6000*50,:);


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


%% here all ready 
%facenet_train_set  facenet_train_label facenet_gallery facenet_gallery_label facenet_probe_c facenet_probe_o1 facenet_probe_o2  facenet_probe_o3
