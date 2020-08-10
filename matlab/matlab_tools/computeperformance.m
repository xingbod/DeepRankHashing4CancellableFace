function [EER, mTSR, mFAR, mFRR, mGAR,EERthreshold] = computeperformance(gen, imp, STEP)
%COMPUTEPERFORMANCE Computes performances of algorithm.
%   COMPUTEPERFORMANCE(GEN, IMP, THRES) computes the performance of the
%   algorithm in terms of its Genuine Acceptance Rate (GAR), False
%   Acceptance Rate (FAR), False Rejection Rate (FRR) and verification rate
%   where both FAR and FRR are set to be as close to each other as
%   possible. The scores of genuine and imposters, GEN and IMP, can be in
%   numeric (i.e. integer, floating-point numbers) or bitstrings form. The
%   step/interval for the iteration used to obtain the threhold value, 
%   STEP, is determined by the user. The default value for STEP is 0.1.


% Find the starting and terminating values for the iteration to find the 
% optimal threshold value.

start = min(gen);
if min(imp) < start
    start = min(imp);
end
stop = max(imp);
if max(gen) > stop
    stop = max(gen);
end

% Calculate TSR, FAR, FRR and GAR for each threshold value.

j = 1;
%display('FAR:                 FRR:')
%disp('start');
%disp(start);
%disp('stop');
%disp(stop);
%user_entry = input('prompt');
% disp('start');
% disp(start);
% disp('STEP');
% disp(STEP);
% disp('stop');
% disp(stop);

[mTSR, mFAR, mFRR, mGAR] =calculateverificationrate(start, gen, imp);

%STEP = (stop - start)/100;
threshold_score= start+STEP : STEP : stop;
for i = start+STEP : STEP : stop
    [TSR, FAR, FRR, GAR] = calculateverificationrate(i, gen, imp);
    y(j) = FRR;
    x(j) = FAR;
    z(j) = GAR;
    s(j) = TSR;
    j = j+1;
%     fprintf('%6.2f \t %6.2f \t %6.2f \t %6.2f \n', i, FAR, FRR, TSR)
    mTSR = [mTSR TSR];
    mFAR = [mFAR FAR];
    mFRR = [mFRR FRR];
    mGAR = [mGAR GAR];
end

% Find the optimal threshold where FAR and FRR is the closest with each
% other.

[val, ind] = min(abs(y-x));

% Display performance result.

fprintf('-------------------------------------\n')
fprintf('\t\tPerformance Result\n')
fprintf('-------------------------------------\n')
fprintf('Verification Rate\t\t: %6.2f %%\n', s(ind));
fprintf('Genuine Acceptance Rate\t: %6.2f %%\n', z(ind));
fprintf('False Acceptance Rate\t: %6.2f %%\n', x(ind));
fprintf('False Rejection Rate\t: %6.2f %%\n', y(ind));
fprintf('Equal Error Rate\t\t: %6.2f %%\n', (y(ind)+x(ind))/2);
fprintf('Threshold\t\t: %6.2f %\r\n', threshold_score(ind));
EERthreshold=threshold_score(ind);
EER = (y(ind)+x(ind))/2;
