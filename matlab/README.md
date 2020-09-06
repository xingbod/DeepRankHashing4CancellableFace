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


enroll_query_search
enroll_query_search_fusion
are for searching, without mixing and demixing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
enroll_query_lfw_iom_id
enroll_query_lfw_iom_id_fusion
are for identification under mixing
%%%%%%%%%%%%%%%%%%%%%%%%%
