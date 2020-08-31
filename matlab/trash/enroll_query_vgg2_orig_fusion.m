function enroll_query_vgg2_orig_fusion(feat_path,feat_path2,filename_path)
% hashcode_path e.g. res50_lfw_feat_dIoM_512x2.csv
% filename_path e.g. lresnet100e_ir_lfw_name.txt
% e.g. enroll_query_iom lresnet100e_ir_lfw_feat_dIoM_512x2.csv  lresnet100e_ir_lfw_name.txt
addpath('../');
addpath('matlab_tools')
addpath_recurse('BLUFR')
addpath_recurse('btp')
addpath('k_reciprocal_re_ranking')

%%
k1 = 20;
k2 = 6;
lambda = 0.3;
measure = 'Euclidean';
%%

Descriptor_orig1 = importdata("../embeddings/"+feat_path);
Descriptor_orig2 = importdata("../embeddings/"+feat_path2);
Descriptor_orig = [Descriptor_orig1 Descriptor_orig2]; % fusion 

ds = "VGG2";
[hash_facenet_probe_c,hash_facenet_probe_o1,hash_facenet_probe_o2,hash_facenet_probe_o3,hash_facenet_gallery,facenet_probe_label_c,facenet_probe_label_o1,facenet_probe_label_o2,facenet_probe_label_o3, facenet_gallery_label] = generateDataset(ds,Descriptor_orig,fid_lfw_name);

%%
[perf] = evaluate_searching(hash_facenet_probe_c,hash_facenet_probe_o1,hash_facenet_probe_o2,hash_facenet_probe_o3,hash_facenet_gallery,facenet_probe_label_c,facenet_probe_label_o1,facenet_probe_label_o2,facenet_probe_label_o3, facenet_gallery_label,measure);
perf = [perf 0 0];

fid=fopen('logs/log_vgg2_fusion.txt','a');
fwrite(fid,feat_path+"_"+feat_path2+" ");
fclose(fid)
dlmwrite('logs/log_vgg2_fusion.txt', perf, '-append');
end