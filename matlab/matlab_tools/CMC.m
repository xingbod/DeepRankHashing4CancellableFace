function [max_rank,rec_rates] =  CMC(match_similarity,num_test,num_train)

% Compute the cosine similarity score between the test samples.
% match_dist=NxM�ľ�������NΪ����������test set����,MΪѵ��������training set����
% match_dist =1./(pdist2( probe_c,gallery,  'euclidean')+0.001);

% num_testΪ����������ǩlabels������num_classΪ�ܵ�����ǩ������num_trainΪע����gallery��ѵ�����еı�ǩ����
%�������������ע���ѵ������ƥ������ƶȾ���
% num_test = probe_label_c;
% num_train = gallery_label;
num_class = unique(num_train);
true_labels = zeros(length(num_test),length(num_class));
for i=1:length(num_test)
    for j=1:length(num_class)
        [x,y]=find(num_class(j)==num_train);
        %ѡȡƥ��̶ȵ���ֵ
        label_distances(i,j) = median(match_similarity(i,y));
        if num_test(i)==num_class(j)
            true_labels(i,j)=1;
        end
    end
end

%����CMC
max_rank = length(num_class);

%Rankȡֵ��Χ
ranks = 1:max_rank;

%����
label_distances_sort = zeros(length(num_test),length(num_class));
true_labels_sort = zeros(length(num_test),length(num_class));
for i=1:length(num_test)
    [label_distances_sort(i,:), ind] = sort(label_distances(i,:),'descend');
    true_labels_sort(i,:) =  true_labels(i,ind);
end
%����
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