function [denoised,Sigma2,P,SNR_gain] = denoise_recursive_tensor(data,window,varargin)
% MP-PCA based denoising of multidimensional (tensor structured) data.
%
% Usage is free but please cite Olesen, JL, Ianus, A, Østergaard, L,
% Shemesh, N, Jespersen, SN. Tensor denoising of multidimensional MRI data.
% Magn Reson Med. 2022; 1- 13. doi:10.1002/mrm.29478
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input variables. 
%
% data: data with noise window: window to pass over data with. Typically
% voxels. 
% 
% window = [5 5] would process patches of data with dimensions 5x5x...
% 
% varargin: is specified as name-value pairs (i.e. ...,'mask',mask,...)
%             indices: determines tensor-struture of patches. For instance
%             for data with 5 indices with the three first being voxels,
%             indices = {1:3 4 5} will combine the voxels into one index
%             and retain the others so that each patch is denoised as a
%             three-index tensor -- indices = {1:2 3 4 5} would denoise
%             each patch as a four-index tensor and so on. It defaults to
%             combining the voxel/window indices and sorting according to
%             index dimensionality in ascending order, since this appears
%             to be optimal in most cases. mask: if a logical mask is
%             specified, locations were the sliding window contains no
%             "true" voxels are skipped. opt_shrink: uses optimal shrinkage
%             if true (default is false) sigma: specifies a know value for
%             the noise sigma rather than estimating it using
%             MP-distribution
%
% Output: 
% 
% denoised: the denoised data 
% 
% Sigma2: estimated noise variance 
% 
% P: estimated number signal components 
% 
% SNR_gain: an estimate of the estimated gain in signal-to-noise ratio.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reshape data to have all voxels and measurement indices along two indices
dims = size(data);
dims_vox = dims(1:length(window));
if numel(dims_vox)==1 % to be compatible with matlab "size" in case window only has one element
    dims_vox(2) = 1;
end
num_vox = prod(dims_vox);
data = reshape(data,num_vox,[]);

% determine default index ordering (same order as input data and all of
% them with window indices combined in one)
vox_indices = 1:length(window);
mod_indices = length(window)+1:length(dims);

% get optional input
options.indices = cat(2,{vox_indices},num2cell(mod_indices));
options.mask = true(dims_vox);
options.center_assign = false;
options.stride = ones(1,length(window));
options.test = false;
for n = 1:2:length(varargin)
    options.(varargin{n}) = varargin{n+1};
end
indices = options.indices;
stride = reshape(options.stride,1,[]);
assert(all(size(options.mask)==dims_vox),'mask dimensions do not match data dimensions');

% dimensions of X array
window = reshape(window,1,[]);
array_size = [window dims(length(window)+1:end)];
array_size = cat(2,array_size,ones(1,max(cell2mat(indices))-length(dims)));

% index addition vector (indices of voxels within sliding window reltive to
% corner index)
window_subs = cell(1,length(window));
[window_subs{:}] = ind2sub(window,1:prod(window));
index_increments = -1 + sub2ind(dims,window_subs{:});

% permutation order in accordance with provided indices
permute_order = cell2mat(indices);
permute_order = cat(2,permute_order,setdiff(1:length(array_size),permute_order));

% size after reshaping in accordance with provided indices
new_size = zeros(1,numel(indices)+1);
for n = 1:numel(indices)
    new_size(n) = prod(array_size(indices{n}));
end
new_size(end) = prod(array_size)/prod(new_size(1:end-1));
new_size(new_size==1) = [];
new_size(end+1) = 1;

% pre-allocate
denoised = zeros(size(data),'like',data);
count = zeros(num_vox,1);
Sigma2 = zeros(num_vox,1);
P = zeros(num_vox,1);

% loop over window positions and denoise
for i = 1:num_vox
    % check if sliding window is within bounds
    index_vector = get_index_vector(dims_vox,i); % indices of window corner
    if any(index_vector-1+window>dims_vox) % if outside bounds, move window to next position
        continue
    end

    % simply skip to next position if this one does not correspond to
    % correct stride
    if any(rem(index_vector-1,stride))
        continue
    end
    
    % indices of voxels within window
    vox_indices = i + index_increments;
    
    % skip if no voxels in mask are included
    maskX = options.mask(vox_indices);
    if nnz(maskX)==0
        continue
    end
    if options.center_assign % or skip if centre voxel is not included
        center_subs = num2cell(ceil(window/2));
        center_ind = sub2ind(window,center_subs{:});
        if ~maskX(center_ind)
            continue
        end
    end
    
    % Create data matrix
    X = reshape(data(vox_indices,:),array_size);
    X = permute(X,permute_order);
    X = reshape(X,new_size);

    % denoise X
    [X,sigma2,p] = denoise_array_recursive_tensor(X,'num_inds',min(length(size(X)),numel(indices)),varargin{:});

    X = reshape(X,array_size(permute_order));
    X = ipermute(reshape(X,array_size(permute_order)),permute_order);

    % assign
    if options.center_assign % only use center voxel
        X = reshape(X,numel(vox_indices),[]);
        X = X(center_ind,:);
        vox_indices = vox_indices(center_ind);
    end
    denoised(vox_indices,:) = denoised(vox_indices,:) + reshape(X,numel(vox_indices),[]);
    count(vox_indices) = count(vox_indices) + 1;
    Sigma2(vox_indices) = Sigma2(vox_indices) + sigma2;
    P(vox_indices,1:length(p)) = P(vox_indices,:) + p;
end

% assign to skipped voxels
skipped_vox = count==0;
denoised(skipped_vox,:) = data(skipped_vox,:); % assign original data to skipped voxels
Sigma2(skipped_vox) = nan;
P(skipped_vox) = nan;

% divided by number of times each voxel has been visited to get average
count(skipped_vox) = 1; % to not divide by zero
denoised = denoised./count;
Sigma2 = Sigma2./count;
P = P./count;

% estimate SNR gain according to removed variance
if size(P,2)==1
    SNR_gain = sqrt(prod(new_size)./(P.^2+sum((new_size(1:end-1)-P).*P,2)));
else
    SNR_gain = sqrt(prod(new_size)./(prod(P,2)+sum((new_size(1:end-1)-P).*P,2)));
end

% adjust output to match input dimensions
denoised = reshape(denoised,dims);
Sigma2 = reshape(Sigma2,dims_vox);
P = reshape(P,[dims_vox size(P,2)]);
SNR_gain = reshape(SNR_gain,dims_vox);
end


function index_vector = get_index_vector(dims_vox,i)
index_vector = cell(1,numel(dims_vox));
[index_vector{:}] = ind2sub(dims_vox,i);
index_vector = cell2mat(index_vector);
end