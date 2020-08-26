
veriFarPoints = [0, kron(10.^(-8:-1), 1:9), 1]; % FAR points for face verification ROC plot
osiFarPoints = [0, kron(10.^(-4:-1), 1:9), 1]; % FAR points for open-set face identification ROC plot
rankPoints = [1:10, 20:10:100]; % rank points for open-set face identification CMC plot
reportVeriFar = 0.001; % the FAR point for verification performance reporting
reportOsiFar = 0.01; % the FAR point for open-set identification performance reporting
reportRank = 1; % the rank point for open-set identification performance reporting


numTrials = 1;

numVeriFarPoints = length(veriFarPoints);
iom_score_avg_VR = zeros(numTrials, numVeriFarPoints); % verification rates of the 10 trials
iom_score_avg_veriFAR = zeros(numTrials, numVeriFarPoints); % verification false accept rates of the 10 trials

numOsiFarPoints = length(osiFarPoints);
numRanks = length(rankPoints);
iom_score_avg_DIR = zeros(numRanks, numOsiFarPoints, numTrials); % detection and identification rates of the 10 trials
iom_score_avg_osiFAR = zeros(numTrials, numOsiFarPoints); % open-set identification false accept rates of the 10 trials

%% Get the FAR or rank index where we report performance.
[~, veriFarIndex] = ismember(reportVeriFar, veriFarPoints);
[~, osiFarIndex] = ismember(reportOsiFar, osiFarPoints);
[~, rankIndex] = ismember(reportRank, rankPoints);



%*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&opts.insightface_model
%%%
model = opts.facenet_model;
db_data.X=facenet_gallery';
[all_code, ~] = IoM(db_data, opts, model);
hashed_code_facenet_gallery=all_code.Hx';

db_data.X=facenet_probe_c';
[all_code, ~] = IoM(db_data, opts, model);
hashed_code_facenet_probe_c=all_code.Hx';

db_data.X=facenet_probe_o1';
[all_code, ~] = IoM(db_data, opts, model);
hashed_code_facenet_probe_o1=all_code.Hx';

db_data.X=facenet_probe_o2';
[all_code, ~] = IoM(db_data, opts, model);
hashed_code_facenet_probe_o2=all_code.Hx';

db_data.X=facenet_probe_o3';
[all_code, ~] = IoM(db_data, opts, model);
hashed_code_facenet_probe_o3=all_code.Hx';


%%%
model = opts.insightface_model;
db_data.X=insightface_gallery';
[all_code, ~] = IoM(db_data, opts, model);
hashed_code_insightface_gallery=all_code.Hx';

db_data.X=insightface_probe_c';
[all_code, ~] = IoM(db_data, opts, model);
hashed_code_insightface_probe_c=all_code.Hx';

db_data.X=insightface_probe_o1';
[all_code, ~] = IoM(db_data, opts, model);
hashed_code_insightface_probe_o1=all_code.Hx';

db_data.X=insightface_probe_o2';
[all_code, ~] = IoM(db_data, opts, model);
hashed_code_insightface_probe_o2=all_code.Hx';

db_data.X=insightface_probe_o3';
[all_code, ~] = IoM(db_data, opts, model);
hashed_code_insightface_probe_o3=all_code.Hx';



% Compute the cosine similarity score between the test samples.
facenet_score_c =1-(pdist2( hashed_code_facenet_gallery,hashed_code_facenet_probe_c,  'jaccard'));
facenet_score_o1 =1-(pdist2( hashed_code_facenet_gallery,hashed_code_facenet_probe_o1,  'jaccard'));
facenet_score_o2 =1-(pdist2( hashed_code_facenet_gallery,hashed_code_facenet_probe_o2,  'jaccard'));
facenet_score_o3 =1-(pdist2( hashed_code_facenet_gallery,hashed_code_facenet_probe_o3,  'jaccard'));
% Compute the cosine similarity score between the test samples.
insightface_score_c =1-(pdist2( hashed_code_insightface_gallery,hashed_code_insightface_probe_c,  'jaccard'));
insightface_score_o1 =1-(pdist2( hashed_code_insightface_gallery,hashed_code_insightface_probe_o1,  'jaccard'));
insightface_score_o2 =1-(pdist2( hashed_code_insightface_gallery,hashed_code_insightface_probe_o2,  'jaccard'));
insightface_score_o3 =1-(pdist2( hashed_code_insightface_gallery,hashed_code_insightface_probe_o3,  'jaccard'));


facenet_match_similarity =1-(pdist2( hashed_code_facenet_probe_c, hashed_code_facenet_gallery, 'jaccard'));
insightface_match_similarity =1-(pdist2( hashed_code_insightface_probe_c, hashed_code_insightface_gallery, 'jaccard'));


score_c =(facenet_score_c+insightface_score_c)./2;
score_o1 =(facenet_score_o1+insightface_score_o1)./2;
score_o2 =(facenet_score_o2+insightface_score_o2)./2;
score_o3 =(facenet_score_o3+insightface_score_o3)./2;
match_similarity =(facenet_match_similarity+insightface_match_similarity)./2;


% Compute the cosine similarity score between the test samples.
% Evaluate the verification performance.
[iom_score_avg_VR(1,:), iom_score_avg_veriFAR(1,:)] = EvalROC(score_c, facenet_gallery_label, facenet_probe_label_c, veriFarPoints);
% CMC close set
[iom_score_avg_rank,iom_score_avg_rec_rates] = CMC(match_similarity,facenet_probe_label_c,facenet_gallery_label);
% Evaluate the open-set identification performance.
[iom_score_avg_DIR(:,:,1), iom_score_avg_osiFAR(1,:)] = OpenSetROC(score_o1, facenet_gallery_label, facenet_probe_label_o1, osiFarPoints );
[iom_score_avg_DIR(:,:,2), iom_score_avg_osiFAR(2,:)] = OpenSetROC(score_o2, facenet_gallery_label, facenet_probe_label_o2, osiFarPoints );
[iom_score_avg_DIR(:,:,3), iom_score_avg_osiFAR(3,:)] = OpenSetROC(score_o3, facenet_gallery_label, facenet_probe_label_o3, osiFarPoints );


save('data/iom_score_avg_veriFAR.mat','iom_score_avg_veriFAR');
save('data/iom_score_avg_max_rank.mat','iom_score_avg_rank');
save('data/iom_score_avg_VR.mat','iom_score_avg_VR');
save('data/iom_score_avg_rec_rates.mat','iom_score_avg_rec_rates');
save('data/iom_score_avg_osiFAR.mat','iom_score_avg_osiFAR');
save('data/iom_score_avg_DIR.mat','iom_score_avg_DIR');
