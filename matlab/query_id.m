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
        correct_ret = correct_ret+1;
    end
end

tar = correct_ret/size(facenet_probe_label_c,2)% 97.41 %     0.8874