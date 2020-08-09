%% This demo shows how to apply a supervised learning method for 
% the BLUFR benchmark and report results. Also, it shows how to learn
% optimal parameters using the development set.
close all; clear; clc;

feaFile = '../data/lfw.mat'; % Mat file storing extracted features. Replace this with your own feature.
configFile = '../config/lfw/blufr_lfw_config.mat'; % configuration file for this evaluation
outDir = '../result/'; % output directory
outMatFile = [outDir, 'result_lfw_supervised.mat']; % output mat file
outLogFile = [outDir, 'result_lfw_supervised.txt']; % output text file

veriFarPoints = [0, kron(10.^(-8:-1), 1:9), 1]; % FAR points for face verification ROC plot
osiFarPoints = [0, kron(10.^(-4:-1), 1:9), 1]; % FAR points for open-set face identification ROC plot
rankPoints = [1:10, 20:10:100]; % rank points for open-set face identification CMC plot
reportVeriFar = 0.001; % the FAR point for verification performance reporting
reportOsiFar = 0.01; % the FAR point for open-set identification performance reporting
reportRank = 1; % the rank point for open-set identification performance reporting

tic;
fprintf('Load data...\n\n');
load(configFile);

%% Load your own features here. The features should be extracted according
% to the order of the imageList in the configFile. It is 13233xd for the 
% LFW database where d is the feature dimensionality.
load(feaFile, 'Descriptors');

% You may apply the sqrt transform if the feature is histogram based.
Descriptors = sqrt(double(Descriptors));

%% Get the training and test data of the development set.
trainX = Descriptors(devTrainIndex, :);
trainLabels = labels(devTrainIndex);
testX = Descriptors(devTestIndex, :);
testLabels = labels(devTestIndex);

%% Do a grid search of optimal parameters.
[pcaDims, lambda] = GridSearch(trainX, trainLabels, testX, testLabels);

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
    
    % Get the training and test data of the t'th trial.
    trainX = Descriptors(trainIndex{t}, :);
    trainLabels = labels(trainIndex{t});
    testX = Descriptors(testIndex{t}, :);
    testLabels = labels(testIndex{t});
    
    % Learn a PCA subspace. Note that if you apply a learning based dimension 
    % reduction, it must be performed with the training data of each trial. 
    % It is not allowed to learn and reduce the dimensionality of features
    % with the whole data beforehand and then do the 10-trial evaluation.
    W = PCA(trainX);
    
    % Transform the training and test data into the learned PCA subspace of pcaDims dimensions.
    trainX = trainX * W(:, 1 : pcaDims);
    testX = testX * W(:, 1 : pcaDims);

    % Select classes which have at least two images.
    hst = hist(trainLabels, 1 : max(trainLabels));
    classIndex = find(hst >= 2);
    [sampleIndex, trainLabels] = ismember(trainLabels, classIndex); % class labels are continuously relabeled starting from 1
    trainLabels = trainLabels(sampleIndex);
    trainX = trainX(sampleIndex,:);
    numTrainSamples = length(trainLabels);

    % Construct the discriminant response matrix
    Y = zeros(numTrainSamples, length(classIndex));
    Y( sub2ind(size(Y), 1 : numTrainSamples, trainLabels') ) = 1;
    
    % Learn a linear subspace by ridge regression. You can replace this with
    % your own supervised learning algorithm here.
    W = (trainX' * trainX + lambda * numTrainSamples * eye(pcaDims)) \ (trainX' * Y);

    % Transform the test data into the learned subspace.
    testX = testX * W;
    
    % Normlize each row to unit length. If you do not have this function,
    % do it manually.
    testX = normr(testX);
    
    % Compute the cosine similarity score between the test samples.
    score = testX * testX';    
    
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

clear Descriptors trainX testX Y W score

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

%% Plot the face verification ROC curve.
figure; semilogx(meanVeriFAR * 100, fusedVR, 'LineWidth', 2);
xlim([0,100]); ylim([0,100]); grid on;
xlabel('False Accept Rate (%)');
ylabel('Verification Rate (%)');
title('Face Verification ROC Curve');

%% Plot the open-set face identification ROC curve at the report rank.
figure; semilogx(meanOsiFAR * 100, fusedDIR(rankIndex,:), 'LineWidth', 2);
xlim([0,100]); ylim([0,100]); grid on;
xlabel('False Accept Rate (%)');
ylabel('Detection and Identification Rate (%)');
title(sprintf('Open-set Identification ROC Curve at Rank %d', reportRank));

%% Plot the open-set face identification CMC curve at the report FAR.
figure; semilogx(rankPoints, fusedDIR(:,osiFarIndex), 'LineWidth', 2);
xlim([0,100]); ylim([0,100]); grid on;
xlabel('Rank');
ylabel('Detection and Identification Rate (%)');
title( sprintf('Open-set Identification CMC Curve at FAR = %g%%', reportOsiFar*100) );

%% Save the results to a mat file. If your result is among the top 10 
% results (ranked by verification rates at FAR=0.1%) maintained in our
% project page, please send this mat file to us so that we can update the
% top 10 results to include your algorithm's performance.
save(outMatFile, 'reportVeriFar', 'reportOsiFar', 'reportRank', 'reportVR', 'reportDIR', ...
    'meanVeriFAR', 'fusedVR', 'meanOsiFAR', 'fusedDIR', 'rankPoints', 'rankIndex', 'osiFarIndex');
