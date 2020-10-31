function run_single_learning(embpath,backbone,DS,LUT)
% backbone = "ResNet50";%InceptionResNetV2
% DS = "IJBC";run_single("ResNet50","LFW") run_single_learning("embeddings_dl/ResNet50_lfw_feat_dlIoM_512x2.csv","embeddings_dl/ResNet50_lfw_name_512x2.txt","LFW")
% enroll_query_search("ResNet50_lfw_feat_dlIoM_512x2","ResNet50_lfw_name_512x2txt",measure,DS,remark);
%
%measure = "Hamming";
%remark = "deep_learning_iom";
%% enroll_query_search(featpath,namepath,measure,DS,remark);
%
%enroll_query_iom_id(featpath,namepath,measure,DS,remark);
%
%


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

remark = "learning_vgg2_diomh";
measure = "Hamming";
for m = [32 64 128 256 512]%64 128 256
    for q = [8]
        tic
        enroll_query_search( embpath+"/"+backbone+"_"+embed+"_feat_randomIoM_0_LUT_"+num2str(LUT)+"_"+num2str(m)+"x"+num2str(q)+".csv", embpath+'/'+backbone+'_'+embed+'_name_randomIoM_1_LUT_'+num2str(LUT)+'_'+num2str(m)+"x"+num2str(q)+'.txt',measure,DS,remark);
        toc
    end
end

end