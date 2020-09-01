backbone = "InceptionResNetV2";

measure = "Euclidean";
remark = "deep_orig_feat";
enroll_query_search("embeddings_0831/"+backbone+"_lfw_feat.csv", 'embeddings/ResNet50_lfw_name.txt',measure,"LFW",remark);
% enroll_query_search("embeddings_0831/ResNet50_VGG2_feat.csv", 'embeddings_0831/ResNet50_VGG2_name.txt',measure,"VGG2",remark);
enroll_query_search("embeddings_0831/"+backbone+"_ijbc_feat.csv", 'embeddings_0831/ResNet50_ijbc_name.txt',measure,"IJBC",remark);

remark = "random_iom";
measure = "Hamming";

for m = [64 128 256 512]
    for q = [8]
        tic
        enroll_query_search("embeddings_0831/"+backbone+"_lfw_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings/ResNet50_lfw_name.txt',measure,"LFW",remark);
        %         enroll_query_search("embeddings_0831/ResNet50_VGG2_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_0831/ResNet50_VGG2_name_64x8.txt',measure,"VGG2",remark);
        enroll_query_search("embeddings_0831/"+backbone+"_ijbc_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_0831/ResNet50_ijbc_name_64x8.txt',measure,"IJBC",remark);
        toc
    end
end

for m = [512]
    for q = [2 4 16]
        tic
        enroll_query_search("embeddings_0831/"+backbone+"_lfw_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings/ResNet50_lfw_name.txt',measure,"LFW",remark);
        %         enroll_query_search("embeddings_0831/ResNet50_VGG2_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_0831/ResNet50_VGG2_name_64x8.txt',measure,"VGG2",remark);
        enroll_query_search("embeddings_0831/"+backbone+"_ijbc_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_0831/ResNet50_ijbc_name_64x8.txt',measure,"IJBC",remark);
        toc
    end
end

%%
remark = "random_iom_identification";
measure = "Hamming";
for m = [64 128 256 512]
    for q = [8]
        tic
        enroll_query_iom_id("embeddings_0831/"+backbone+"_lfw_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings/ResNet50_lfw_name.txt',measure,"LFW",remark)
        %         enroll_query_iom_id("embeddings_0831/ResNet50_VGG2_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_0831/ResNet50_VGG2_name_64x8.txt',measure,"VGG2",remark)
        enroll_query_iom_id("embeddings_0831/"+backbone+"_ijbc_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_0831/ResNet50_ijbc_name_64x8.txt',measure,"IJBC",remark)
        toc
    end
end

for m = [512]
    for q = [2 4 16]
        tic
        enroll_query_iom_id("embeddings_0831/ResNet50_lfw_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings/ResNet50_lfw_name.txt',measure,"LFW",remark)
        %         enroll_query_iom_id("embeddings_0831/ResNet50_VGG2_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_0831/ResNet50_VGG2_name_64x8.txt',measure,"VGG2",remark)
        enroll_query_iom_id("embeddings_0831/ResNet50_ijbc_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_0831/ResNet50_ijbc_name_64x8.txt',measure,"IJBC",remark)
        toc
    end
end

