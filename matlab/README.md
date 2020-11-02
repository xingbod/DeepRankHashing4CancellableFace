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


# Unlinkability and revocability

_unlinkability.m_

Using this script to generate mated non-mated scores.

_revocablity.m_

Using this script to generate genuine scores.




```bash
measure = "Hamming";
remark = "learning_incepv2";
enroll_query_search("embeddings_dl/InceptionResNetV2_VGG2_feat_randomIoM_0_LUT_0_512x8.csv", "embeddings_dl/InceptionResNetV2_VGG2_name_randomIoM_0_LUT_0_512x8.txt",measure,"VGG2",remark);

```


# 20201028
```bash
remark = "learning_xception_diomh";                      
DS= "VGG2";                                            
q=8;                                                                                                                      
m=512;     
remark = "learning_xceptio_diomh";  
enroll_query_search( "embeddings_inresv2/Xception_VGG2_feat_randomIoM_0_LUT_0_512x8.csv", "embeddings_inresv2/Xception_VGG2_name_randomIoM_0_LUT_0_512x8.txt",measure,DS,remark);

remark = "learning_xception_diomh";                      
DS= "LFW";    
measure = "Hamming";
enroll_query_search( "embeddings_inresv2/Xception_lfw_feat_randomIoM_0_LUT_0_512x8.csv", "embeddings_inresv2/Xception_lfw_name_randomIoM_0_LUT_0_512x8.txt",measure,DS,remark);


remark = "random_ResNet50_diomh";                      
DS= "VGG2";    
measure = "Hamming";
enroll_query_search( "embeddings_inresv2/ResNet50_VGG2_feat_randomIoM_1_LUT_0_512x8.csv", "embeddings_inresv2/ResNet50_VGG2_name_randomIoM_0_LUT_0_512x8.txt",measure,DS,remark);


remark = "random_ResNet50_LUT_diomh";                      
DS= "LFW";    
measure = "Euclidean";
enroll_query_search( "embeddings_inresv2/ResNet50_lfw_feat_randomIoM_1_LUT_3_512x8.csv", "embeddings_inresv2/ResNet50_lfw_name_randomIoM_1_LUT_3_512x8.txt",measure,DS,remark);

remark = "learning_ResNet50_LUT_diomh";                      
DS= "LFW";    
measure = "Hamming";
enroll_query_search( "embeddings_inresv2/ResNet50_lfw_feat_randomIoM_0_LUT_3_512x8.csv", "embeddings_inresv2/ResNet50_lfw_name_randomIoM_0_LUT_3_512x8.txt",measure,DS,remark);

```

# 20201031

```bash

run_single("embeddings_inresv2","InceptionResNetV2","VGG2",0)
run_single_learning("embeddings_inresv2","InceptionResNetV2","VGG2",0)

```


# 20201102

```bash
remark = "learning_VGG2_fusion_diomh";                      
DS= "VGG2";    
measure = "Hamming";
(feat_path,feat_path2,filename_path,measure,ds,remark)
enroll_query_search_fusion("embeddings_inresv2/Xception_lfw_feat_randomIoM_0_LUT_0_512x8.csv","embeddings_inresv2/InceptionResNetV2_lfw_feat_randomIoM_0_LUT_0_512x8.csv")

```
