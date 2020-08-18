addpath('k_reciprocal_re_ranking')
%% re-ranking setting
k1 = 20;
k2 = 6;
lambda = 0.3;
%% Euclidean
%dist_eu = pdist2(galFea', probFea');
my_pdist2 = @(A, B) sqrt( bsxfun(@plus, sum(A.^2, 2), sum(B.^2, 2)') - 2*(A*B'));
dist_eu = my_pdist2(galFea', probFea');
[CMC_eu, map_eu, ~, ~] = evaluation(dist_eu, label_gallery, label_query, [], []);

fprintf(['The IDE (' netname ') + Euclidean performance:\n']);
fprintf(' Rank1,  mAP\n');
fprintf('%5.2f%%, %5.2f%%\n\n', CMC_eu(1) * 100, map_eu(1)*100);

%% Euclidean + re-ranking
query_num = size(probFea, 2);
dist_eu_re = re_ranking( [probFea galFea], 1, 1, query_num, k1, k2, lambda);
[CMC_eu_re, map_eu_re, ~, ~] = evaluation(dist_eu_re, label_gallery, label_query, cam_gallery, cam_query);

fprintf(['The IDE (' netname ') + Euclidean + re-ranking performance:\n']);
fprintf(' Rank1,  mAP\n');
fprintf('%5.2f%%, %5.2f%%\n\n', CMC_eu_re(1) * 100, map_eu_re(1)*100);


%%% query phase
% facenet_probe_o3
% facenet_probe_label_o3
correct_ret=0;
incorrect_ret = 0;
for i =1:size(facenet_probe_label_c,2)
    i
    query_sample = dec2bin( hash_facenet_probe_c(i,:)-1,q)-'0';
    query_bin =reshape(query_sample',1,numel(gallery_sample));

    dist = [];
    for j=1:length(mixing_facenet_gallery)
        gallery_bin =  mixing_facenet_gallery(j,:);
        retrieved_id = bitxor(gallery_bin,query_bin);
        dist = [dist pdist2(retrieved_id,identifiers(facenet_gallery_label(j),:),'Hamming')];
    end
    [row column]=find(dist==min(dist(:)));
    if mode(facenet_gallery_label(column)) == facenet_probe_label_c(i)
        correct_ret = correct_ret+1 
    end
end

correct_ret/size(facenet_probe_label_c,2)