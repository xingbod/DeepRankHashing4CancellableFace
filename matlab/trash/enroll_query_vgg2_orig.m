function enroll_query_vgg2_orig(hashcode_path,filename_path)
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

Descriptors = importdata("../embeddings/"+hashcode_path);

ds = "VGG2";
[hash_facenet_probe_c,hash_facenet_probe_o1,hash_facenet_probe_o2,hash_facenet_probe_o3,hash_facenet_gallery,facenet_probe_label_c,facenet_probe_label_o1,facenet_probe_label_o2,facenet_probe_label_o3, facenet_gallery_label] = generateDataset(ds,Descriptor_orig,fid_lfw_name);

%%
[perf] = evaluate_searching(hash_facenet_probe_c,hash_facenet_probe_o1,hash_facenet_probe_o2,hash_facenet_probe_o3,hash_facenet_gallery,facenet_probe_label_c,facenet_probe_label_o1,facenet_probe_label_o2,facenet_probe_label_o3, facenet_gallery_label,measure);
perf = [perf 0 0];
fid=fopen('logs/log_vgg2.txt','a');
fwrite(fid,hashcode_path+" ");
fclose(fid)
dlmwrite('logs/log_vgg2.txt', perf, '-append');
end