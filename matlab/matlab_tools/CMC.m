function [max_rank,rec_rates] =  CMC(match_similarity,num_test,num_train)

% Compute the cosine similarity score between the test samples.
% match_dist=NxM的矩阵，其中N为测试样本（test set）数,M为训练样本（training set）数
% match_dist =1./(pdist2( probe_c,gallery,  'euclidean')+0.001);

% num_test为测试样本标签labels向量，num_class为总的类别标签向量，num_train为注册在gallery或训练集中的标签向量
%处理测试样本与注册的训练样本匹配的相似度矩阵
% num_test = probe_label_c;
% num_train = gallery_label;
num_class = unique(num_train);
true_labels = zeros(length(num_test),length(num_class));
for i=1:length(num_test)
    for j=1:length(num_class)
        [x,y]=find(num_class(j)==num_train);
        %选取匹配程度的中值
        label_distances(i,j) = median(match_similarity(i,y));
        if num_test(i)==num_class(j)
            true_labels(i,j)=1;
        end
    end
end

%生成CMC
max_rank = length(num_class);

%Rank取值范围
ranks = 1:max_rank;

%排序
label_distances_sort = zeros(length(num_test),length(num_class));
true_labels_sort = zeros(length(num_test),length(num_class));
for i=1:length(num_test)
    [label_distances_sort(i,:), ind] = sort(label_distances(i,:),'descend');
    true_labels_sort(i,:) =  true_labels(i,ind);
end
%迭代
rec_rates = zeros(1,max_rank);
tmp = 0;
for i=1:max_rank
    tmp = tmp + sum(true_labels_sort(:,i));
    rec_rates(1,i)=tmp/length(num_test);
end

% semilogx(1:max_rank,rec_rates* 100, 'LineWidth', 2);
% xlim([0,max_rank]); ylim([98,100]); grid on;
% xlabel('Rank');
% ylabel('Recognition Rate');
% title('Close-set C Face Verification CMC Curve');
% x_formatstring = '%6.1f';
% % Here's the code.
% xtick = get(gca, 'xtick');
% for i = 1:length(xtick)
%     xticklabel{i} = sprintf(x_formatstring, xtick(i));
% end
% set(gca, 'xticklabel', xticklabel);

end