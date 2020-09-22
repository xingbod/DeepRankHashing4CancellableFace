# Content
In ths document, the usage of the evauation matlab script will be described.

# Evaluate original deep features accuracy
```bash
measure = "Euclidean";
remark = "deep_orig_feat";
enroll_query_search("embeddings_0831/ResNet50_lfw_feat.csv", "embeddings_0831/ResNet50_lfw_name.txt",measure,"LFW",remark);
enroll_query_search("embeddings_0831/InceptionResNetV2_lfw_feat.csv", "embeddings_0831/ResNet50_lfw_name.txt",measure,"LFW",remark);
enroll_query_search("embeddings_0831/ResNet50_lfw_feat.csv", "embeddings_0831/ResNet50_lfw_name.txt",measure,"LFW",remark);

```

# Fusion

IJBC 

```bash
measure = "Hamming";
remark = "fusion_learn_DIoMH";
ds = "IJBC"
enroll_query_iom_id_fusion("embeddings_dl/ResNet50_ijbc_feat_dlIoM_512x8.csv","embeddings_dl/InceptionResNetV2_ijbc_feat_dlIoM_512x8.csv","embeddings_dl/ResNet50_ijbc_name_dl_256x8.txt",measure,ds,remark)
enroll_query_iom_id_fusion("embeddings_dl/ResNet50_ijbc_feat_dlIoM_512x8.csv","embeddings_dl/Xception_ijbc_feat_dlIoM_512x8.csv","embeddings_dl/ResNet50_ijbc_name_dl_256x8.txt",measure,ds,remark)
enroll_query_iom_id_fusion("embeddings_dl/Xception_ijbc_feat_dlIoM_512x8.csv","embeddings_dl/InceptionResNetV2_ijbc_feat_dlIoM_512x8.csv","embeddings_dl/ResNet50_ijbc_name_dl_256x8.txt",measure,ds,remark)
```

VGG2

```bash
measure = "Hamming";
remark = "fusion_learn_DIoMH";
ds = "VGG2"
enroll_query_iom_id_fusion("embeddings_dl/ResNet50_VGG2_feat_dlIoM_512x8.csv","embeddings_dl/InceptionResNetV2_VGG2_feat_dlIoM_512x8.csv","embeddings_dl/ResNet50_VGG2_name_dl_256x8.txt",measure,ds,remark)

enroll_query_iom_id_fusion("embeddings_dl/ResNet50_VGG2_feat_dlIoM_512x8.csv","embeddings_dl/Xception_VGG2_feat_dlIoM_512x8.csv","embeddings_dl/ResNet50_VGG2_name_dl_256x8.txt",measure,ds,remark)

enroll_query_iom_id_fusion("embeddings_dl/Xception_VGG2_feat_dlIoM_512x8.csv","embeddings_dl/InceptionResNetV2_VGG2_feat_dlIoM_512x8.csv","embeddings_dl/ResNet50_VGG2_name_dl_256x8.txt",measure,ds,remark)

```
VGG2 random DIoMH 
```bash
measure = "Hamming";
remark = "fusion_learn_DIoMH";
ds = "IJBC"
run_single("embeddings_0831","ResNet50","IJBC")  
run_single("embeddings_0831","Xception","IJBC")  
run_single("embeddings_0831","InceptionResNetV2","IJBC")  
```

VGG2 learning cancellable DIoMH 
```bash
run_single_learning_all("embeddings_dl","ResNet50","VGG2")
run_single_learning_all("embeddings_dl","InceptionResNetV2","VGG2")
run_single_learning_all("embeddings_dl","Xception","VGG2")
```

enroll_query_search
enroll_query_search_fusion
are for searching, without mixing and demixing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
enroll_query_lfw_iom_id
enroll_query_lfw_iom_id_fusion
are for identification under mixing
%%%%%%%%%%%%%%%%%%%%%%%%%
