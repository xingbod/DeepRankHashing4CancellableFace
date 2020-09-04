function run_single_learning(featpath,namepath,DS)

% backbone = "ResNet50";%InceptionResNetV2
% DS = "IJBC";run_single("ResNet50","LFW") run_single("embeddings_dl","ResNet50","LFW")

measure = "Hamming";
remark = "deep_learning_iom";
enroll_query_search(featpath,namepath,measure,DS,remark);

enroll_query_iom_id(featpath,namepath,measure,DS,remark);




end