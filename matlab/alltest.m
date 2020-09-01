% enroll_query_iom2('ResNet50_lfw_feat_dIoM_64x8.csv',  'ResNet50_lfw_name.txt')
% enroll_query_iom2('ResNet50_lfw_feat_dIoM_128x8.csv',  'ResNet50_lfw_name.txt')
% enroll_query_iom2('ResNet50_lfw_feat_dIoM_256x8.csv',  'ResNet50_lfw_name.txt')
% enroll_query_iom2('ResNet50_lfw_feat_dIoM_512x8.csv',  'ResNet50_lfw_name.txt')
%
% enroll_query_iom2('allloss_learning_iom_ResNet50_lfw_feat_512x2.csv',  'allloss_learning_iom_ResNet50_lfw_name_512x2.txt')
% enroll_query_iom2('allloss_learning_iom_ResNet50_lfw_feat_512x4.csv',  'allloss_learning_iom_ResNet50_lfw_name_512x2.txt')
% enroll_query_iom2('allloss_learning_iom_ResNet50_lfw_feat_512x8.csv',  'allloss_learning_iom_ResNet50_lfw_name_512x2.txt')
% enroll_query_iom2('allloss_learning_iom_ResNet50_lfw_feat_512x16.csv',  'allloss_learning_iom_ResNet50_lfw_name_512x2.txt')
%
% enroll_query_iom2('learning_iom_ResNet50_lfw_feat_64x8.csv',  'learning_iom_ResNet50_lfw_name_64x8.txt')
% enroll_query_iom2('learning_iom_ResNet50_lfw_feat_128x8.csv',  'learning_iom_ResNet50_lfw_name_64x8.txt')
% enroll_query_iom2('learning_iom_ResNet50_lfw_feat_256x8.csv',  'learning_iom_ResNet50_lfw_name_64x8.txt')
% enroll_query_iom2('learning_iom_ResNet50_lfw_feat_512x8.csv',  'learning_iom_ResNet50_lfw_name_64x8.txt')
%
% enroll_query_iom2('ResNet50_lfw_feat_dIoM_64x2.csv',  'ResNet50_lfw_name.txt')
% enroll_query_iom2('ResNet50_lfw_feat_dIoM_128x2.csv',  'ResNet50_lfw_name.txt')
% enroll_query_iom2('ResNet50_lfw_feat_dIoM_256x2.csv',  'ResNet50_lfw_name.txt')
% enroll_query_iom2('ResNet50_lfw_feat_dIoM_512x2.csv',  'ResNet50_lfw_name.txt')
%
%
% enroll_query_orig('ResNet50_lfw_feat.csv',  'ResNet50_lfw_name.txt')
% enroll_query_orig('InceptionResNetV2_lfw_feat.csv',  'InceptionResNetV2_lfw_name.txt')
% enroll_query_orig('lresnet100e_ir_lfw_feat.csv',  'lresnet100e_ir_lfw_name.txt')
%
%
% enroll_query_orig_fusion('ResNet50_lfw_feat.csv','InceptionResNetV2_lfw_feat.csv','ResNet50_lfw_name.txt')
% enroll_query_orig_fusion('ResNet50_lfw_feat.csv','lresnet100e_ir_lfw_feat.csv','ResNet50_lfw_name.txt')
% enroll_query_orig_fusion('lresnet100e_ir_lfw_feat.csv','InceptionResNetV2_lfw_feat.csv','ResNet50_lfw_name.txt')
%
% for m = [64 128 256 512]
%     for q = [2 4 8 16]
%         enroll_query_hashing("ResNet50_lfw_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv",  'ResNet50_lfw_name.txt')
%     end
% end
%
% for m = [64 128 256 512]
%     for q = [2 4 8 16]
%         enroll_query_hashing("lresnet100e_ir_lfw_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv",  'ResNet50_lfw_name.txt')
%     end
% end
%
% for m = [64 128 256 512]
%     for q = [2 4 8 16]
%         enroll_query_hashing("InceptionResNetV2_lfw_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv",  'ResNet50_lfw_name.txt')
%     end
% end
% %%
measure = "Euclidean";
ds = "LFW";
remark = "deep_orig_feat";
enroll_query_search("embeddings_0831/ResNet50_lfw_feat.csv", 'embeddings/ResNet50_lfw_name.txt',measure,"LFW",remark)
enroll_query_search("embeddings_0831/ResNet50_VGG2_feat.csv", 'embeddings_0831/ResNet50_VGG2_name.txt',measure,"VGG2",remark)
enroll_query_search("embeddings_0831/ResNet50_ijbc_feat.csv", 'embeddings_0831/ResNet50_ijbc_name.txt',measure,"IJBC",remark)

remark = "random_iom";
measure = "Hamming";

for m = [64 128 256 512]
    for q = [8]
        enroll_query_search("embeddings_0831/ResNet50_lfw_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings/ResNet50_lfw_name.txt',measure,"LFW",remark)
        enroll_query_search("embeddings_0831/ResNet50_VGG2_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_0831/ResNet50_VGG2_name_64x8.txt',measure,"VGG2",remark)
        enroll_query_search("embeddings_0831/ResNet50_ijbc_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_0831/ResNet50_ijbc_name_64x8.txt',measure,"IJBC",remark)
    end
end

for m = [512]
    for q = [2 4 16]
        enroll_query_search("embeddings_0831/ResNet50_lfw_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings/ResNet50_lfw_name.txt',measure,"LFW",remark)
        enroll_query_search("embeddings_0831/ResNet50_VGG2_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_0831/ResNet50_VGG2_name_64x8.txt',measure,"VGG2",remark)
        enroll_query_search("embeddings_0831/ResNet50_ijbc_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_0831/ResNet50_ijbc_name_64x8.txt',measure,"IJBC",remark)
    end
end

