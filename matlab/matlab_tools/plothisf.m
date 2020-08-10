function plothisf(client,imposter,flag,bitlength,maxscore,maxsum)
%function plothis(client,imposter,flag)
% plothisf(gen,imp,'bit',1,1,2000)
if nargin <4
   bitlength = 0; 
   maxscore=1;
   maxsum=2000;
end    

collection_dist_correct=client;
collection_dist_incorrect=imposter;

if  strcmp(flag,'fm')
    maxi = max(collection_dist_incorrect);
    incr = maxi/length(collection_dist_correct); % Fix a step for X axis values for the histogram
    X_vect = [incr:incr:maxi]; % X axis values for the histogram
    % Normalization of the distance histogram, to get the different users peak at 0.5
    [Y,I]=max(hist(collection_dist_incorrect,X_vect));
    peak = X_vect(I);
    collection_dist_correct = collection_dist_correct/(2*peak);
    collection_dist_incorrect = collection_dist_incorrect/(2*peak);  
    X_vect = X_vect/(2*peak);
elseif strcmp(flag,'bit')    
    X_vect = [0:0.01:bitlength];
%   X_vect = [0:0.0008:bitlength];


else
    disp('Error! only fm or bit is allowed')
% break
end    

collection_dist_correct2 = zeros(1,length(X_vect)+3);
collection_dist_incorrect2 = zeros(1,length(X_vect)+3);

collection_dist_correct2(1,1:length(X_vect)) = hist(collection_dist_correct,X_vect);
collection_dist_incorrect2(1,1:length(X_vect)) = hist(collection_dist_incorrect,X_vect);

% Compute the mean and variance for same/different users histograms
collection_dist_correct2(1,length(X_vect)+1) = mean(collection_dist_correct);
collection_dist_incorrect2(1,length(X_vect)+1) = mean(collection_dist_incorrect);

collection_dist_correct2(1,length(X_vect)+2) = var(collection_dist_correct);
collection_dist_incorrect2(1,length(X_vect)+2) = var(collection_dist_incorrect);

index_save=1;
index_eigenspace=1;

X_tot{index_save,index_eigenspace} = X_vect;
collection_dist_correct_tot{index_save,index_eigenspace} = collection_dist_correct2;
collection_dist_incorrect_tot{index_save,index_eigenspace} = collection_dist_incorrect2;

%--------------------------------------------------------------------------
l=1;
%From libstatistics_displaygraphics
Y1 = collection_dist_correct_tot{l,end}(1:end-3); % get the Y axis values of the different bins of the histogram for same users
Y2 = collection_dist_incorrect_tot{l,end}(1:end-3); % get the Y axis values of the different bins of the histogram for different users
Y1 = Y1*max(Y2)/max(Y1); % normalize the height of both curves
X_vect = X_tot{l,end}; % get the X axis values

mean1(1) = collection_dist_correct_tot{l,end}(end-2); % get the mean distance for same users
mean2(1) = collection_dist_incorrect_tot{l,end}(end-2); % get the mean distance for different users
var1(1) = collection_dist_correct_tot{l,end}(end-1); % get the distance variance for same users
var2(1) = collection_dist_incorrect_tot{l,end}(end-1); % get the distance variance for different users

FRR = collection_dist_correct_tot{l,end}(end); % FRR at FAR = 0%
threshold =collection_dist_incorrect_tot{l,end}(end); % corresponding threshold

% Plot the histogram and display some useful informations
%plot(X_vect,Y1,'r-',X_vect,Y2,'b-.',[mean1,mean1],[0,max(Y1)],'r-',[mean2,mean2],[0,max(Y2)],'b--','LineWidth',1.5);


plot(X_vect,Y1,'r-',X_vect,Y2,'b-.','LineWidth',1.5);

xlabel('Normalized No.genuine pairs count')
ylabel('Frequency')
title('No. genuine pairs collected for Genuine & imposter attempts')

set(gcf,'color',[1 1 1])
legend(strcat('Genuine (mean: ',num2str(mean1),' | ','var: ',...
    num2str(var1),')'),strcat('Imposter (mean: ',num2str(mean2),' | ','var: ',num2str(var2),')'));

axis([0 maxscore 0 maxsum]);

% axis([0 1 0 1000])