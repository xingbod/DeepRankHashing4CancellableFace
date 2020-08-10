======================================================
AGREEMENT ON THE USE OF IFO Hashing ALGORITHM AND ANY GENERATED DATA

I agree:

1. To cite [1] in any paper of mine or my collaborators that makes any use of the codes or data generated from these codes.
2. To use the codes and generated data for research purposes only.
3. Not to provide the codes or generated data to third parties without the notice of authors and this agreement.

[1] Y. L. Lai, Z. Jin, A. B. J. Teoh, B. M. Goi, W. S. Yap, T. Y. Chai, C. Rathgeb. “Cancellable Iris Template Generation based on Indexing-First-One”, Pattern Recognition, 64, pp. 105-117, 2017. 


=== To use this code === 
1. Select input parameters value for (tau,P,K, m) 
2. Generate random permutation matrix (e.g. tokens) for IFO hashing: [ PermuteMatx ] = Generate_PermMx( InputSample,P,m)
3. Generate IFO hashed code: [IFOCode,PermuteMatx] = IFO(InputSample,tau,PermuteMatx,P, K,m)
4. Matching between different IFO hashed codes:  [Score, No_of_Collision] = IFO_matching(IFOcode1, IFOcode2)

*This code may not be optimized,only for research purpose.