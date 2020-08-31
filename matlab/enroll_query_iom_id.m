function enroll_query_iom_id(hashcode_path,filename_path,measure,ds,remark)
% hashcode_path e.g. res50_lfw_feat_dIoM_512x2.csv
% filename_path e.g. lresnet100e_ir_lfw_name.txt
% e.g. enroll_query_lfw_iom_id lresnet100e_ir_lfw_feat_dIoM_512x2.csv  lresnet100e_ir_lfw_name.txt
addpath('../');
addpath('matlab_tools')
addpath_recurse('BLUFR')
addpath_recurse('btp')
addpath('k_reciprocal_re_ranking')

%%
k1 = 20;
k2 = 6;
lambda = 0.3;
% measure = 'Hamming';
% ds = "LFW";
log_path = "logs/log_"+ds+"_"+remark+".log";

%%

Descriptor_orig = importdata("../embeddings/"+hashcode_path);
fid_lfw_name=importdata("../embeddings/" + filename_path);



[hash_facenet_probe_c,hash_facenet_probe_o1,hash_facenet_probe_o2,hash_facenet_probe_o3,hash_facenet_gallery,facenet_probe_label_c,facenet_probe_label_o1,facenet_probe_label_o2,facenet_probe_label_o3, facenet_gallery_label] = generateDataset(ds,Descriptor_orig,fid_lfw_name);


[perf] = evaluate_searching_mixdemix(hash_facenet_probe_c,hash_facenet_probe_o1,hash_facenet_probe_o2,hash_facenet_probe_o3,hash_facenet_gallery,facenet_probe_label_c,facenet_probe_label_o1,facenet_probe_label_o2,facenet_probe_label_o3, facenet_gallery_label);
perf = [perf 0 0];
fid=fopen(log_path,'a');
fwrite(fid,hashcode_path+" ");
fclose(fid)
dlmwrite(log_path, perf, '-append');
end