run_single_learning_all("embeddings_dl","Xception","VGG2")
run_single_learning_all("embeddings_dl","ResNet50","VGG2")
run_single_learning_all("embeddings_dl","InceptionResNetV2","VGG2")

measure = "Hamming";
remark = "fusion_learn_DIoMH";
ds = "VGG2"
enroll_query_iom_id_fusion("embeddings_dl/ResNet50_VGG2_feat_dlIoM_512x8.csv","embeddings_dl/InceptionResNetV2_VGG2_feat_dlIoM_512x8.csv","embeddings_dl/ResNet50_VGG2_name_dl_256x8.txt",measure,ds,remark)

enroll_query_iom_id_fusion("embeddings_dl/ResNet50_VGG2_feat_dlIoM_512x8.csv","embeddings_dl/Xception_VGG2_feat_dlIoM_512x8.csv","embeddings_dl/ResNet50_VGG2_name_dl_256x8.txt",measure,ds,remark)

enroll_query_iom_id_fusion("embeddings_dl/Xception_VGG2_feat_dlIoM_512x8.csv","embeddings_dl/InceptionResNetV2_VGG2_feat_dlIoM_512x8.csv","embeddings_dl/ResNet50_VGG2_name_dl_256x8.txt",measure,ds,remark)


run_single("embeddings_0831","Xception","VGG2")
run_single("embeddings_0831","ResNet50","VGG2")
run_single("embeddings_0831","InceptionResNetV2","VGG2")
