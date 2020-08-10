function [biocode] = bioencoding(onetemplate,randomString,m)

%m=5; % 6 bits per word
n=10240; % iriscode 10240bits
% biocodes n/m 1707 bits
l= floor(n/m);


biocode=zeros(1,l);
for i=1:l
    hist= onetemplate((i-1)*m+1:i*m);
    location =sum(power(2,m-find(hist>0)+1-1))+1;
    biocode(i) = randomString(location);
end

end


