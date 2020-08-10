function [TSR, FAR, FRR, GAR] = calculateverificationrate(thr, client, ...
    imposter)
%CALCULATEVERIFICATIONRATE calculates verification rate for a threshold.
%   CALCVERFRATE(THR, CLIENT, IMPOSTER) calculates the verification rate
%   for the threshold value, THR. 
%
%   The output TSR, FAR, FRR and GAR stand for verification rate, false
%   acceptance rate, false rejection rate and genuine acceptance rate,
%   respectively.

lenclient = length(client);
lenImp = length(imposter);

% Calculate FAR

a = find( imposter >= thr);
FA = length(a);
FAR = (FA/lenImp)*100;

% Calculate FRR

b= find(client < thr);
FR = length(b);
FRR = (FR/lenclient)*100;

% Calculate GAR

c= find(client > thr);
GA = length(c);
GAR = (GA/lenclient)*100;

% Calculate TSR

TER = (FA + FR)/(lenclient + lenImp);
TSR = (1 - TER)*100;

