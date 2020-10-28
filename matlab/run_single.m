function run_single(embpath,backbone,DS)
% backbone = "ResNet50";%InceptionResNetV2
% DS = "IJBC";run_single("ResNet50","LFW")
% run_single("embeddings_0831","InceptionResNetV2","LFW") run_single("embeddings_0831","Xception","LFW")
% run_single("embeddings_0831","ResNet50","LFW")
% run_single("embeddings_0831","ResNet50","IJBC")  
% run_single("embeddings_0831","ResNet50","VGG2")
% InceptionResNetV2_lfw_name_randomIoM_0_LUT_0_512x8.txt    
% InceptionResNetV2_lfw_feat_randomIoM_0_LUT_0_512x8

if DS == "IJBC"
    embed = "ijbc";
elseif DS == "VGG2"
    embed = "VGG2";
else DS == "LFW"
    embed = "lfw";
end

% measure = "Cosine";
% remark = "deep_orig_feat";
% enroll_query_search(embpath+"/"+backbone+"_"+embed+"_feat.csv", 'embeddings_0831/'+backbone+'_'+embed+'_name.txt',measure,DS,remark);

remark = "learning_diomh";
measure = "Hamming";
for m = [32 64 128 256 512]%64 128 256
    for q = [8]
        tic
        enroll_query_search( embpath+"/"+backbone+"_"+embed+"_feat_randomIoM_0_LUT_0_"+num2str(m)+"x"+num2str(q)+".csv", embpath+'/'+backbone+'_'+embed+'_name_randomIoM_0_LUT_0_'++num2str(m)+"x"+num2str(q)+'.txt',measure,DS,remark);
        toc
    end
end

% for m = [512]
%     for q = [2 4 16]
%         tic
%         enroll_query_search(embpath+"/"+backbone+"_"+embed+"_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_0831/'+backbone+'_'+embed+'_name.txt',measure,DS,remark);
%         toc
%     end
% end

%
% remark = "random_iom_identification";
% measure = "Hamming";
% for m = [32 64 128 256 512]
%     for q = [8]
%         tic_dl
%         enroll_query_iom_id(embpath+"/"+backbone+"_"+embed+"_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_0831/'+backbone+'_'+embed+'_name.txt',measure,DS,remark);
%         toc
%     end
% end
% 
% for m = [512]
%     for q = [2 4 16]
%         tic
%         enroll_query_iom_id(embpath+"/"+backbone+"_"+embed+"_feat_drIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_0831/'+backbone+'_'+embed+'_name.txt',measure,DS,remark);
%         toc
%     end
% end



end