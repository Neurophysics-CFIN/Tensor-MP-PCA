function [X,sigma2,P] = denoise_array_recursive_tensor(X,varargin)
% input:
% X: array to be denoised

dims = size(X);

% handle inputs
options.opt_shrink = true;
options.subtract_mean = false;
options.num_inds = length(dims); % number of indices to iterate through (typically set to number of indices in X but can be less to omit denoising specific indices)
options.full_sigma2_pass = true;
options.test = false;
for n = 1:2:length(varargin)
    options.(varargin{n}) = varargin{n+1};
end
use_MPPCA = ~isfield(options,'sigma2');
if ~use_MPPCA
    sigma2 = options.sigma2;
    options.use_initial_sigma2_pass = false;
end

%% handle special case of matrix input
if length(dims)==2
    if min(dims)==1 % handle special case where Mp=1
        sigma2 = 0;
        P = 1;
        return
    end

    % subtract mean from X if specified
    if options.subtract_mean
        [X,X_mean] = subtract_mean(X);
    else
        X_mean = 0;
    end
    
    % get singular values and vectors
    [U,S,V] = svd(X,'econ');
    
    % MP cutoff
    if use_MPPCA
        sigma2 = estimate_noise(S,dims);
    end
    
    % apply cutoff
    [U,S,V,P] = discard_noise_components(U,S,V,sigma2);
    
    % optimal shrinkage
    if options.opt_shrink
        S = apply_optimal_shrinkage(U,S,V,sigma2);
    end
    
    % reconstruct X
    X = U*S*V' + X_mean;
    return
end


%% estimate sigma2 from first SVD or make full HOSVD pass to get all singular values for combined sigma2 estimate
if use_MPPCA % estimate noise if not specified

    if options.full_sigma2_pass
        num_SVDs = options.num_inds;
    else
        num_SVDs = 1;
    end
    for n = 1:num_SVDs
        X = reshape(X,dims(n),[]); % i_n-flattening

        if options.subtract_mean
            [X,X_mean{n}] = subtract_mean(X);
        else
            X_mean{n} = 0;
        end

        [U{n},S{n},V{n}] = svd(X,'econ');
        
        [~,P(n)] = estimate_noise(S{n},dims); % get initial individual P estimates

        X = V{n}*S{n}; % prepare X for next iteration (transpose: make current first index the last index without changing ordering of other indices i.e. 123 -> 231)
    end

    [sigma2,P] = combined_noise_estimate(S,dims,P);


else % prepare X so code below is the same whether or not use_MPPCA==true
    n = 1;
    X = reshape(X,dims(n),[]); % i_n-flattening
    
    if options.subtract_mean
        [X,X_mean{n}] = subtract_mean(X);
    else
        X_mean{n} = 0;
    end

    [U{n},S{n},V{n}] = svd(X,'econ');
end


%% do recursive SVD
% reuse calculations for n==1 above
n = 1;
[U{n},S{n},V{n},P(n)] = discard_noise_components(U{n},S{n},V{n},sigma2);

X = V{n}*S{n}; % prepare the remaining part of partially denoised X for next iteration

% continue for remaining indices
for n = 2:options.num_inds
    if P(n-1)==0
        P = P(1:n-1);
        break
    end

    X = reshape(X,dims(n),[]); % i_n-flattening

    if options.subtract_mean
        [X,X_mean{n}] = subtract_mean(X);
    else
        X_mean{n} = 0;
    end

    [U{n},S{n},V{n}] = svd(X,'econ'); % get new singular values and vectors of partially denoised X

    [U{n},S{n},V{n},P(n)] = discard_noise_components(U{n},S{n},V{n},sigma2);
    
    if options.opt_shrink && n==options.num_inds % apply optimal shrinkage at last iteration
        S{n} = apply_optimal_shrinkage(U{n},S{n},V{n},sigma2);
    end

    X = V{n}*S{n}; % prepare the remaining part of partially denoised X for next iteration
end

%% reconstruct denoised X
for n = flip(1:length(P)) % backward-pass: remember initially X = V*S from last iteration
    if P(n)==0
        X = zeros(size(U{n},1),size(X,1)) + X_mean{n};
    else
        X = U{n}*reshape(X,[],P(n))' + X_mean{n};
    end
end
X = reshape(X,dims); % invert flattening
P = cat(2,P,zeros(1,options.num_inds-length(P)));



function [sigma2,P] = estimate_noise(S,dims)
M = size(S,1);
N = prod(dims)/M; % need to use original dimensions of X if no components are discarded
vals2 = diag(S).^2;
P = (0:length(vals2)-1)'; % all possible values for number of signal components
sigma2_estimates = cumsum(vals2,'reverse') ./ (M-P) ./ (N-P); % sigma2 estimate as a function of number of signal components
cutoff_estimates = sigma2_estimates * (sqrt(M)+sqrt(N))^2; % upper cutoff of MP distribution as a function of number of signal components
P = -1 + find(vals2<cutoff_estimates,1); % every singular value below cutoff is a noise component
if isempty(P) % handle special case when no noise components are found
    P = length(vals2);
    sigma2 = 0;
else
    sigma2 = sigma2_estimates(P+1);
end
if P==0 && min(M,N)==1 % special case
    P = 1;
end



function [sigma2,P] = combined_noise_estimate(S,dims,P)
sigma2 = 0;
denominator = 0;
for n = 1:length(S)
    M = size(S{n},1);
    N = prod(dims)/M;
    vals2 = diag(S{n}).^2;
    sigma2 = sigma2 + sum(vals2(P(n)+1:end));
    denominator = denominator + (M-P(n))*(N-P(n));
end
sigma2 = sigma2/denominator;
for n = 1:length(S)
    M = size(S{n},1);
    N = prod(dims)/M;
    cutoff = sigma2 * (sqrt(M)+sqrt(N))^2;
    P(n) = nnz(diag(S{n}).^2>cutoff);
end



function [U,S,V,P] = discard_noise_components(U,S,V,sigma2)
M = size(U,1);
N = size(V,1);
cutoff = sigma2 * (sqrt(M)+sqrt(N))^2;
P = nnz(diag(S).^2>cutoff);
U = U(:,1:P);
S = S(1:P,1:P);
V = V(:,1:P);



function S = apply_optimal_shrinkage(U,S,V,sigma2)
if S==0 % if no signal components were found
    return
end
M = size(U,1);
N = size(V,1);
P = size(S,1);
vals2 = diag(S).^2;
vals2 = opt_shrink_frob(vals2,max(M-P,1),max(N-P,1),sigma2);
S = diag(real(sqrt(vals2)));



function [X,X_mean] = subtract_mean(X)
[M,N] = size(X);
if M<N
    X_mean = mean(X,2);
else
    X_mean = mean(X,1);
end
X = X - X_mean;

