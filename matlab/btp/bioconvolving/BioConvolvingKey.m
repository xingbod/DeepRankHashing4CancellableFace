function [key]=generateBioConvolvingKey(w,n)
%w= number of partitions. We consider the w as the size of the key
%n= number of features of the biometric data
%% dj random numbers

%random numbers between 1 and 99
%     r = (99-1).*rand(w-1,1) + 1;
%     d=sort(r);
%
%     %% Putting 0 at index 1 and 100 in the last position
%     d(2:end+1)=d;
%     d(1)=1;
%     d(end+1)=100;
%
%     key=round((d/100)*n);
%
%     if key(w)==n ||key(w)==0
%         key=generateBioConvolvingKey(w,n);
%     end

if w==25
  p=sort(randperm(n));
else
  p=sort(randperm(n-1,w));
end
key=[1];
for i=1:w-1
  key=[key,p(i)];
end
key=[key,n];
end