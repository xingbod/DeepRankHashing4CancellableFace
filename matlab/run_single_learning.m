function run_single_learning(featpath,namepath,DS)

% backbone = "ResNet50";%InceptionResNetV2
% DS = "IJBC";run_single("ResNet50","LFW") run_single_learning("../embeddings_dl/ResNet50_lfw_feat_dlIoM_512x2.csv","../embeddings_dl/ResNet50_lfw_name_512x2.txt","LFW")
% enroll_query_search("ResNet50_lfw_feat_dlIoM_512x2","ResNet50_lfw_name_512x2txt",measure,DS,remark);

measure = "Hamming";
remark = "deep_learning_iom";
enroll_query_search(featpath,namepath,measure,DS,remark);

enroll_query_iom_id(featpath,namepath,measure,DS,remark);




end