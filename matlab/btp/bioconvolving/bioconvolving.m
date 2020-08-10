function [transformed_data] = bioconvolving(biometric_data,key)

transformed_data=[];
if isempty(key)
    key=BioConvolvingKey(2,length(biometric_data(1,:)));
end

for sample=1:length(biometric_data(:,1))
    c=0;
    for i=1:length(key)-1
        if i==1
            size=key(i+1)-key(i);
            c=biometric_data(sample,1:(size+key(i)));
        else
            n=key(i)+1;
            size=key(i+1)-key(i);
            a=biometric_data(sample,n:(size+key(i)));
            c=conv(c,a);
        end
    end
    transformed_data=[transformed_data;c];
end
end