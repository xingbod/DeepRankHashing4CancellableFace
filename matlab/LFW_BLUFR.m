function [reportVeriFar, reportVR,reportRank, reportOsiFar, reportDIR] = LFW_BLUFR(varargin)
%LFW_BLUFR �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
ip = inputParser;
%�������趨Ĭ��ֵ��������ָ���Ǳ���������ǿ�ѡ�����ȡ�
ip.addRequired('Descriptors')
ip.addParamValue('measure','Hamming');
ip.parse(varargin{:});
result=ip.Results;

Descriptors = result.Descriptors;
measure = result.measure;

feaFile = 'BLUFR/data/lfw.mat'; % Mat file storing extracted features
configFile = 'BLUFR/config/lfw/blufr_lfw_config.mat'; % configuration file for this evaluation
outDir = 'BLUFR/result/'; % output directory
outMatFile = [outDir, 'result_lfw_pca.mat']; % output mat file
outLogFile = [outDir, 'result_lfw_pca.txt']; % output text file

veriFarPoints = [0, kron(10.^(-8:-1), 1:9), 1]; % FAR points for face verification ROC plot
osiFarPoints = [0, kron(10.^(-4:-1), 1:9), 1]; % FAR points for open-set face identification ROC plot
rankPoints = [1:10, 20:10:100]; % rank points for open-set face identification CMC plot
reportVeriFar = 0.001; % the FAR point for verification performance reporting
reportOsiFar = 0.01; % the FAR point for open-set identification performance reporting
reportRank = 1; % the rank point for open-set identification performance reporting

pcaDims = 400; % PCA dimensions.

tic;
fprintf('Load data...\n\n');
load(configFile);

%% Load your own features here. The features should be extracted according
% to the order of the imageList in the configFile. It is 13233xd for the
% LFW database where d is the feature dimensionality.


% You may apply the sqrt transform if the feature is histogram based.
% Descriptors = sqrt(double(Descriptors));

numTrials = length(testIndex);

numVeriFarPoints = length(veriFarPoints);
VR = zeros(numTrials, numVeriFarPoints); % verification rates of the 10 trials
veriFAR = zeros(numTrials, numVeriFarPoints); % verification false accept rates of the 10 trials

numOsiFarPoints = length(osiFarPoints);
numRanks = length(rankPoints);
DIR = zeros(numRanks, numOsiFarPoints, numTrials); % detection and identification rates of the 10 trials
osiFAR = zeros(numTrials, numOsiFarPoints); % open-set identification false accept rates of the 10 trials

%% Get the FAR or rank index where we report performance.
[~, veriFarIndex] = ismember(reportVeriFar, veriFarPoints);
[~, osiFarIndex] = ismember(reportOsiFar, osiFarPoints);
[~, rankIndex] = ismember(reportRank, rankPoints);

fprintf('Evaluation with 10 trials.\n\n');

%% Evaluate with 10 trials.
for t = 1 : numTrials
    fprintf('Process the %dth trial...\n\n', t);
    
    % Get the training data of the t'th trial.
    %     X = Descriptors(trainIndex{t}, :);
    
    % Learn a PCA subspace. Note that if you apply a learning based dimension
    % reduction, it must be performed with the training data of each trial.
    % It is not allowed to learn and reduce the dimensionality of features
    % with the whole data beforehand and then do the 10-trial evaluation.
    %     W = PCA(X,pcaDims);
    
    % Get the test data of the t'th trial
    X = Descriptors(testIndex{t}, :);
    
    % Transform the test data into the learned PCA subspace of pcaDims dimensions.
    %     X = X * W(:, 1 : pcaDims);
    
    % Normlize each row to unit length. If you do not have this function,
    % do it manually.
    X = normr(X);
    
    % Compute the cosine similarity score between the test samples.
    %     score = X * X';
    score = 1- pdist2(X,X,measure);
    
    % Get the class labels for the test data of the development set.
    testLabels = labels(testIndex{t});
    
    % Evaluate the verification performance.
    [VR(t,:), veriFAR(t,:)] = EvalROC(score, testLabels, [], veriFarPoints);
    
    % Get the gallery and probe index in the test set
    [~, gIdx] = ismember(galIndex{t}, testIndex{t});
    [~, pIdx] = ismember(probIndex{t}, testIndex{t});
    
    % Evaluate the open-set identification performance.
    [DIR(:,:,t), osiFAR(t,:)] = OpenSetROC( score(gIdx, pIdx), testLabels(gIdx), testLabels(pIdx), osiFarPoints );
    
    fprintf('Verification:\n');
    fprintf('\t@ FAR = %g%%: VR = %g%%.\n', reportVeriFar*100, VR(t, veriFarIndex)*100);
    
    fprintf('Open-set Identification:\n');
    fprintf('\t@ Rank = %d, FAR = %g%%: DIR = %g%%.\n\n', reportRank, reportOsiFar*100, DIR(rankIndex, osiFarIndex, t)*100);
end

clear Descriptors X W score

%% Average over the 10 trials, and compute the standard deviation.
meanVeriFAR = mean(veriFAR);
meanVR = mean(VR);
stdVR = std(VR);
reportMeanVR = meanVR(veriFarIndex);
reportStdVR = stdVR(veriFarIndex);

meanOsiFAR = mean(osiFAR);
meanDIR = mean(DIR, 3);
stdDIR = std(DIR, 0, 3);
reportMeanDIR = meanDIR(rankIndex, osiFarIndex);
reportStdDIR = stdDIR(rankIndex, osiFarIndex);

%% Get the mu - sigma performance measures
fusedVR = ( meanVR - stdVR ) * 100;
reportVR = (reportMeanVR - reportStdVR) * 100;
fusedDIR = ( meanDIR - stdDIR ) * 100;
reportDIR = (reportMeanDIR - reportStdDIR) * 100;

testTime = toc;
fprintf('Evaluation time: %.0f seconds.\n\n', testTime);

%% Display the benchmark performance and output to the log file.
str = sprintf('Verification:\n');
str = sprintf('%s\t@ FAR = %g%%: VR = %.2f%%.\n', str, reportVeriFar*100, reportVR);

str = sprintf('%sOpen-set Identification:\n', str);
str = sprintf('%s\t@ Rank = %d, FAR = %g%%: DIR = %.2f%%.\n\n', str, reportRank, reportOsiFar*100, reportDIR);



fprintf('The fused (mu - sigma) performance:\n\n');
fprintf('%s', str);
fout = fopen(outLogFile, 'wt');
fprintf(fout, '%s', str);
fclose(fout);




end

