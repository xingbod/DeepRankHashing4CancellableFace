% run_single_learning_all("embeddings_dl","Xception","IJBC")
% run_single_learning_all("embeddings_dl","ResNet50","IJBC")
% run_single_learning_all("embeddings_dl","InceptionResNetV2","IJBC")

measure = "Hamming";
remark = "fusion_learn_DIoMH";
ds = "IJBC"
enroll_query_iom_id_fusion("embeddings_dl/ResNet50_IJBC_feat_dlIoM_512x8.csv","embeddings_dl/InceptionResNetV2_IJBC_feat_dlIoM_512x8.csv","embeddings_dl/ResNet50_IJBC_name_dl_256x8.txt",measure,ds,remark)

enroll_query_iom_id_fusion("embeddings_dl/ResNet50_IJBC_feat_dlIoM_512x8.csv","embeddings_dl/Xception_IJBC_feat_dlIoM_512x8.csv","embeddings_dl/ResNet50_IJBC_name_dl_256x8.txt",measure,ds,remark)

enroll_query_iom_id_fusion("embeddings_dl/Xception_IJBC_feat_dlIoM_512x8.csv","embeddings_dl/InceptionResNetV2_IJBC_feat_dlIoM_512x8.csv","embeddings_dl/ResNet50_IJBC_name_dl_256x8.txt",measure,ds,remark)

run_single("embeddings_0831","Xception","IJBC")
run_single("embeddings_0831","ResNet50","IJBC")
run_single("embeddings_0831","InceptionResNetV2","IJBC")
