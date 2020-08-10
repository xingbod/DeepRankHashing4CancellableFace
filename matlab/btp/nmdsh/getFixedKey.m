function key=getFixedKey(cancelableFunction, keySize)
if strcmp(cancelableFunction,'Interpolation')    
    load('InterpolationKey.mat','key');
    key=key(1:keySize);
elseif strcmp(cancelableFunction,'BioHashing')
    load('BioHashingKey.mat','key');
    key=key(1:keySize,1:keySize);
elseif strcmp(cancelableFunction,'BioConvolving')
    p=keySize.nFeatures/keySize.partitions;
    key=[1];
    for i=1:keySize.partitions-1
        if p*i>keySize.nFeatures
            break
        end
        key=[key,round(p*i)];
    end
    key=[key,keySize.nFeatures];
%     key=[0,round(keySize/2),keySize];
elseif strcmp(cancelableFunction,'DoubleSum')
    load('DoubleSumKey.mat','key');
    key=key(1:keySize);
    
    %key=1:2:keySize;
    %key=[key,2:2:keySize];
end
end