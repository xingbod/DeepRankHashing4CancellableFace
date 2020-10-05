function run_single_learning_all(embpath,backbone,DS)

% backbone = "ResNet50";%InceptionResNetV2
% DS = "IJBC";run_single("ResNet50","LFW")
% run_single("embeddings_dl","InceptionResNetV2","LFW")
% run_single("embeddings_dl","ResNet50","LFW")
% run_single("embeddings_dl","ResNet50","IJBC")  
% run_single("embeddings_dl","ResNet50","IJBC")
% run_single_learning_all("embeddings_dl","Xception","IJBC")
% run_single_learning_all("embeddings_dl","Xception","VGG2")
% run_single_learning_all("embeddings_dl","ResNet50","IJBC")
% run_single_learning_all("embeddings_dl","ResNet50","VGG2")
% run_single_learning_all("embeddings_dl","InceptionResNetV2","IJBC")
% run_single_learning_all("embeddings_dl","InceptionResNetV2","VGG2")
% run_single_learning_all("embeddings_0831","ResNet50","IJBC")
% run_single_learning_all("embeddings_0831","InceptionResNetV2","IJBC")
% run_single_learning_all("embeddings_0831","Xception","IJBC")


if DS == "IJBC"
    embed = "ijbc";
elseif DS == "VGG2"
    embed = "VGG2";
else DS == "LFW"
    embed = "lfw";
end
% measure = "Euclidean";
% remark = "learning_DIoMH";
% enroll_query_search(embpath+"/"+backbone+"_"+embed+"_feat.csv", 'embeddings_dl/'+backbone+'_'+embed+'_name_dl_64x8.txt',measure,DS,remark);
% 
% remark = "learning_iom_search";
% measure = "Hamming";
% for m = [64 128 256 512]
%     for q = [8]
%         tic 
%         enroll_query_search(embpath+"/"+backbone+"_"+embed+"_feat_dlIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_dl/'+backbone+'_'+embed+'_name_dl_64x8.txt',measure,DS,remark);
%         toc
%     end
% end
% 
% for m = [512]
%     for q = [2 4 16]
%         tic
%         enroll_query_search(embpath+"/"+backbone+"_"+embed+"_feat_dlIoM_"+num2str(m)+"x"+num2str(q)+".csv", 'embeddings_dl/'+backbone+'_'+embed+'_name_dl_64x8.txt',measure,DS,remark);
%         toc
%     end
% end

%%
remark = "learning_iom_identification";
measure = "Hamming";
for m = [64 128 256 512]
    for q = [8]
        tic
        enroll_query_iom_id(embpath+"/"+backbone+"_"+embed+"_feat_dlIoM_"+num2str(m)+"x"+num2str(q)+".csv",  'embeddings_dl/'+backbone+'_'+embed+'_name_'+num2str(m)+"x"+num2str(q)'.txt',measure,DS,remark);
        toc
    end
end

for m = [512]
    for q = [2 4 16]
        tic
        enroll_query_iom_id(embpath+"/"+backbone+"_"+embed+"_feat_dlIoM_"+num2str(m)+"x"+num2str(q)+".csv",  'embeddings_dl/'+backbone+'_'+embed+'_name_'+num2str(m)+"x"+num2str(q)'.txt',measure,DS,remark);
        toc
    end
end



end