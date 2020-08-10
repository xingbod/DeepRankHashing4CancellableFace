function[IFOCode,PermuteMatx] = IFO(X,tau,PermuteMatx,P, K,m)

sindex=K-tau;

if m==1
   





segment=ones(size(X,1),K);S_PermuteMatx=PermuteMatx{1,1};
for i = 1:P 
        interest_segment= X(:,S_PermuteMatx(i,1:K));
        segment=segment.*interest_segment;
    
end

IFOCode=zeros(size(X,1),1);

 for  j=1:size(X,1)
     D=0;
for k=1:K
    
    if segment(j,k)>D
        D=segment(j,k);
        IFOCode(j)=k;
         if IFOCode(j)>sindex
        IFOCode(j)=mod(IFOCode(j),sindex);
            
        end
    else
  
        IFOCode(j)=IFOCode(j);
   
    end
    
  
end
 end
 
 
 
elseif m>1
   
IFOCode=zeros(size(X,1),m); IFOCode2=zeros(size(X,1),1); 
for ii=1:m
     
    S_PermuteMatx=PermuteMatx{1,ii};


segment=ones(size(X,1),K);
for i = 1:P 
        interest_segment= X(:,S_PermuteMatx(i,1:K));
        segment=segment.*interest_segment;
    
end

 for  j=1:size(X,1)
     D=0;
for k=1:K
    
    if segment(j,k)>D
        D=segment(j,k);
        
        
        IFOCode2(j)=k;
        if IFOCode2(j)>sindex
        IFOCode2(j)=mod(IFOCode2(j),sindex);
            
        end
        
    else 
  
        IFOCode2(j)=IFOCode2(j);
       
   
    end
    

end


 end
    

  IFOCode(:,ii)=IFOCode2(:);IFOCode2=zeros(size(X,1),1); 
    
    
end


end 
    
end 
    
    
   
    
    
    


