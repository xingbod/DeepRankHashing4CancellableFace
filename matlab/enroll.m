close all; clear; clc;
addpath('matlab_tools')
addpath_recurse('BLUFR')
addpath_recurse('btp')
% load('data\lfw_512_insightface_embeddings.mat')
load('data\lfw_label.mat')
load('data\align_lfw_feat.mat')

Descriptors = align_lfw_feat;

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

Nb=400;%400
opts.lambda = 0.5;% 0.5 1 2
opts.beta = 1;% 0.5 0.8 1
opts.K = 16;
opts.topP = 5;
opts.L = 100; % train maximum number of bits
opts.gaussian=1; %1/0=gaussian/laplace
opts.dX=size(Descriptors,2);
opts.softmod=0;
opts.alpha=0.6;
opts.model = random_IoM(opts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = opts.model;
db_data.X=facenet_probe_c';
[all_code, ~] = IoM(db_data, opts, model);
hash_facenet_probe_c=all_code.Hx';

db_data.X=facenet_probe_o1';
[all_code, ~] = IoM(db_data, opts, model);
hash_facenet_probe_o1=all_code.Hx';

db_data.X=facenet_probe_o2';
[all_code, ~] = IoM(db_data, opts, model);
hash_facenet_probe_o2=all_code.Hx';

db_data.X=facenet_probe_o3';
[all_code, ~] = IoM(db_data, opts, model);
hash_facenet_probe_o3=all_code.Hx';


db_data.X=facenet_gallery';
[all_code, ~] = IoM(db_data, opts, model);
hash_facenet_gallery=all_code.Hx';
%%%% generate identifier, dimension same to hash code
m = 100;
q=4;
[identifiers ] = generate_identifier(m,q,6000);
%%% mixing gallery
mixing_facenet_gallery = [];
for i = 1:size(facenet_gallery_label,2)
    gallery_sample = dec2bin( hash_facenet_gallery(i,:)-1,q)-'0';
    gallery_bin =reshape(gallery_sample',1,numel(gallery_sample));
    mixing_facenet_gallery(i,:) = bitxor(gallery_bin,identifiers(facenet_gallery_label(i),:));
end
