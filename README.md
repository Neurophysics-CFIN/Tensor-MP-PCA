# Tensor-MP-PCA
Matlab implementation of MP-PCA based denoising of multidimensional data based on Olesen, JL, Ianus, A, Østergaard, L, Shemesh, N, Jespersen, SN. Tensor denoising of multidimensional MRI data. Magn Reson Med. 2022; 1- 13. doi:10.1002/mrm.29478. Usage is free, but please cite the paper above. The data can also be shared, but due to file size (~40 Gb), must be arranged via file transfer by contacting sune@cfin.au.dk.

Syntax 

[denoised,Sigma2,P,SNR_gain] = denoise_recursive_tensor(data,window,varargin)

MP-PCA based denoising of multidimensional (tensor structured) data.

Input variables. 
data: data with noise window: window to pass over data with. Typically
voxels. 

window = [5 5] would process patches of data with dimensions 5x5x...

varargin: is specified as name-value pairs (i.e. ...,'mask',mask,...)
             indices: determines tensor-struture of patches. For instance
             for data with 5 indices with the three first being voxels,
             indices = {1:3 4 5} will combine the voxels into one index
             and retain the others so that each patch is denoised as a
             three-index tensor -- indices = {1:2 3 4 5} would denoise
             each patch as a four-index tensor and so on. It defaults to
             combining the voxel/window indices and sorting according to
             index dimensionality in ascending order, since this appears
             to be optimal in most cases. mask: if a logical mask is
             specified, locations were the sliding window contains no
             "true" voxels are skipped. opt_shrink: uses optimal shrinkage
             if true (default is false) sigma: specifies a know value for
             the noise sigma rather than estimating it using
             MP-distribution

Output: 
 
denoised: the denoised data 
 
Sigma2: estimated noise variance 
 
P: estimated number signal components 
 
SNR_gain: an estimate of the estimated gain in signal-to-noise ratio.
