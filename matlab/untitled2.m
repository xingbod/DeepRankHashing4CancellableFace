addpath('k_reciprocal_re_ranking')
k1 = 20;
k2 = 6;
lambda = 0.3;

probFea = hash_facenet_probe_c';
galFea = hash_facenet_gallery';
cam_gallery = [];
cam_query = [];
label_gallery = facenet_gallery_label;
label_query = facenet_probe_label_c;
%% Euclidean
%dist_eu = pdist2(galFea', probFea');
% my_pdist2 = @(A, B) sqrt( bsxfun(@plus, sum(A.^2, 2), sum(B.^2, 2)') - 2*(A*B'));
% dist_eu = my_pdist2(galFea', probFea');
dist_eu = pdist2(galFea', probFea','Hamming');
[CMC_eu, map_eu, ~, ~] = evaluation(dist_eu, label_gallery, label_query, cam_gallery, cam_query);

fprintf(['The Euclidean performance:\n']);
fprintf(' Rank1,  mAP\n');
fprintf('%5.2f%%, %5.2f%%\n\n', CMC_eu(1) * 100, map_eu(1)*100);


% original_dist = pdist2(probFea, galFea);
[ final_dist ] = re_ranking2( dist_eu', size(probFea,2), k1, k2, lambda);
