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
for m = [64 128 256 512]
    for q = [2 4 8 16]
        enroll_query_iom_id("ResNet50_lfw_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv",  'ResNet50_lfw_name.txt')
    end
end

for m = [64 128 256 512]
    for q = [2 4 8 16]
        enroll_query_iom_id("lresnet100e_ir_lfw_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv",  'ResNet50_lfw_name.txt')
    end
end

for m = [64 128 256 512]
    for q = [2 4 8 16]
        enroll_query_iom_id("InceptionResNetV2_lfw_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv",  'ResNet50_lfw_name.txt')
    end
end

for m = [64 128 256 512]
    for q = [2 4 8 16]
        enroll_query_iom_fusion_id("ResNet50_lfw_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv","InceptionResNetV2_lfw_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv",  'ResNet50_lfw_name.txt')
    end
end
% 
% enroll_query_iom_fusion(hashcode_path,hashcode_path2,filename_path)