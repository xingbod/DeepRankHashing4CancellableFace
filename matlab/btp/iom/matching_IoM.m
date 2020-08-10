function [similiraty] = matching_IoM(template1,template2)
        C2=abs(template1 - template2);
        CC=find(C2==0);
        totalnumbs=size(CC(),2);
        similiraty= totalnumbs/(length(template1)+length(template2)-totalnumbs);
end