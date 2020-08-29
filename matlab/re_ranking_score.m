function dist_re = re_ranking_score(final_dist,facenet_gallery_label,facenet_probe_label_c,mixing_facenet_gallery,hash_facenet_probe_c, k1, k2, lambda,measure)

cam_gallery = [];
cam_query = [];
label_gallery = facenet_gallery_label;
label_query = facenet_probe_label_c;

% The distance among probes, gallerys can help to improve the accurancy
% -|-
% -|-
% A|B
% C|D
num_probe = size(final_dist,1);
num_gallery = size(final_dist,2);
totallength =num_gallery+num_probe;
new_dist = ones(totallength, totallength);
new_dist(1:num_probe,num_probe+1:totallength) = final_dist;% B
new_dist(num_probe+1:totallength,1:num_probe) = final_dist';% C
final_dist2 = pdist2(mixing_facenet_gallery,mixing_facenet_gallery,measure);
new_dist(num_probe+1:totallength,num_probe+1:totallength) = final_dist2;% D
% new_dist = new_dist - diag(diag(new_dist));
new_dist(1:num_probe,1:num_probe) = pdist2(hash_facenet_probe_c,hash_facenet_probe_c,measure); % A probe vs probe distance

dist_re = re_ranking3( new_dist, num_probe, k1, k2, lambda);

end