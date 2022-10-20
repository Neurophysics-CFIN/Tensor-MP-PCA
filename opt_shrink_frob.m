function vals2 = opt_shrink_frob(vals2,M,N,sigma2)
% DOI: 10.1109/TIT.2017.2653801
% vals2 = vals2/sigma2/N; % rescale
% vals2 = ((vals2-1-M/N).^2-4*M/N)./vals2;
% vals2 = vals2*sigma2*N; % scale back
vals2 = vals2 - 2*(N+M)*sigma2 + (N-M)^2*sigma2^2./vals2;