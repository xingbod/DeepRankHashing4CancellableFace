% 
final_dist = zeros(size(facenet_probe_label_c,2),size(mixing_facenet_gallery,1));
for i = progress(1:size(facenet_probe_label_c,2))
    dist = zeros(1,size(mixing_facenet_gallery,1));
    for j=1: size(mixing_facenet_gallery,1)
        gallery_bin =  mixing_facenet_gallery(j,:);
        retrieved_id = bitxor(gallery_bin,hash_facenet_probe_c(i,:));
        dist(j) = sum(bitxor(retrieved_id,identifiers(facenet_gallery_label(j),:)))/m;
    end
    final_dist(i,:) = dist;

end
% 
% [ final_dist2 ] = re_ranking2( final_dist, k1, k2, lambda)
dist_eu = pdist2(hash_facenet_gallery, hash_facenet_probe_c,'Hamming');
[CMC_eu, map_eu, ~, ~] = evaluation(dist_eu, label_gallery, label_query, cam_gallery, cam_query);

[CMC_eu, map_eu, ~, ~] = evaluation(final_dist', label_gallery, label_query, cam_gallery, cam_query);
 [iom_max_rank,iom_rec_rates] = CMC(1-final_dist,facenet_probe_label_c,facenet_gallery_label);
 [iom_max_rank,iom_rec_rates2] = CMC(1-final_dist2,facenet_probe_label_c,facenet_gallery_label);

fprintf(['The Euclidean performance:\n']);
fprintf(' Rank1,  mAP\n');
fprintf('%5.2f%%, %5.2f%%\n\n', CMC_eu(1) * 100, map_eu(1)*100);

%% Euclidean + re-ranking

[CMC_eu_re, map_eu_re, ~, ~] = evaluation(final_dist2', label_gallery, label_query, cam_gallery, cam_query);

fprintf(['The Euclidean + re-ranking performance:\n']);
fprintf(' Rank1,  mAP\n');
fprintf('%5.2f%%, %5.2f%%\n\n', CMC_eu_re(1) * 100, map_eu_re(1)*100);