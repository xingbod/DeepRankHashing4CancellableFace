function [Score, No_of_Collision] = IFO_matching(template1, template2)


% this code included the filtering process for non-zero elements, follow the IFO paper: 
% https://www.researchgate.net/publication/309873889_Cancellable_Iris_Template_Generation_based_on_Indexing-First-One_Hashing)

   binarize1=logical(template1);
   binarize2=logical(template2);
   nbb=and((binarize1),(binarize2));
   C1=template1(nbb==1); C2=template2(nbb==1); 
   C12=C1-C2;


   totalno_nonzero_elements=size(C12,1);

   CC=(find(C12==0));
   No_of_Collision=size(CC(),1);
        
   Score=No_of_Collision/totalno_nonzero_elements;

end














       
