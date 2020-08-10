function [bins,yys,lambdas,pp]=numericalEigenFunctions2(xx,sig,epsilon)

  if nargin==2
    epsilon=0.1;
  end

  xx=xx(:);
nBins=length(xx)/10;
nBins=min(nBins,50);
[pp,bins]=hist(xx,nBins);
bins=bins(:);
pp=pp/sum(pp);
pOld=pp;
pp=pp+epsilon; % avoid zeros in pdf
pp=pp/sum(pp);

D=distMat([bins(:) bins(:)]/sqrt(2));
%D=dist([bins bins]')/sqrt(2);  % just a hack to get dist to work on 1D vectors
useGaussian=1;

if (useGaussian)
%     W=exp(-0.5*D.^2/sig^2);
    W=exp(-0.5*D.^2/sig^2);
else
    W=0.5*(1-tanh(500*(D-sig)));
end


P=diag(pp);
D3=diag(sum(P*W));
W2=P*W*P;

D2=diag(sum(W2));

L=D2-W2;

% we want to minimize x^T L x / x^T P X
IP=inv(sqrt(P*D3));
L2=IP*L*IP;

[uu,ss,vv]=svd(L2);
yys=IP*uu;
%P=diag(pOld); % measure with original pdf
W2=P*W*P;

D2=diag(sum(W2));

L=D2-W2;
lambdas=diag(yys'*L*yys);
lambdas=lambdas./diag(yys'*P*D3*yys);

%lambdas=diag(ss); %'no binning'
%lambdas=lambdas*(bins(2)-bins(1)).^2;

%keyboard;
