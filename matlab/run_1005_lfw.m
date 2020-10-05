run_single_learning_all("embeddings_dl","Xception","LFW")
run_single_learning_all("embeddings_dl","ResNet50","LFW")
run_single_learning_all("embeddings_dl","InceptionResNetV2","LFW")


measure = "Hamming";
remark = "fusion_learn_DIoMH";
ds = "LFW"
enroll_query_iom_id_fusion("embeddings_dl/ResNet50_LFW_feat_dlIoM_512x8.csv","embeddings_dl/InceptionResNetV2_LFW_feat_dlIoM_512x8.csv","embeddings_dl/ResNet50_LFW_name_dl_256x8.txt",measure,ds,remark)

enroll_query_iom_id_fusion("embeddings_dl/ResNet50_LFW_feat_dlIoM_512x8.csv","embeddings_dl/Xception_LFW_feat_dlIoM_512x8.csv","embeddings_dl/ResNet50_LFW_name_dl_256x8.txt",measure,ds,remark)

enroll_query_iom_id_fusion("embeddings_dl/Xception_LFW_feat_dlIoM_512x8.csv","embeddings_dl/InceptionResNetV2_LFW_feat_dlIoM_512x8.csv","embeddings_dl/ResNet50_LFW_name_dl_256x8.txt",measure,ds,remark)
