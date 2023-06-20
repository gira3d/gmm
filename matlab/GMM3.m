classdef GMM3 < handle
    properties
        means; % stored as 3xn matrix
        covs; % stored as 9xn matrix
        weights; % stored as 1xn vector
        n_components;
        support_size;
    end
    properties (SetAccess = private, Hidden = true)
        objectHandle;
    end
    methods
        function this = GMM3(varargin)

            wrapper = @(x) reshape(x, 1, 3, 3);
            if ((nargin == 2))
                data = varargin{1};
                n_clusters = varargin{2};

                if (size(data,1) ~= 3)
                    data = transpose(data);
                end
                if (size(data,1) ~= 3)
                    error('data dimension incorrect...should be 3xN');
                    return
                end

                [this.weights, this.means, covs, this.objectHandle] = ...
                    gmm_mex('new', data, n_clusters);
                this.n_components = n_clusters;
                this.support_size = size(data,2);

                this.means = this.means;
                this.covs = covs; %cell2mat(cellfun(wrapper, num2cell(covs,1), 'UniformOutput', false)');
                this.weights = this.weights';
            elseif ((nargin == 4))
                this.means = reshape(varargin{1}, 3, length(varargin{1}));
                this.covs = reshape(varargin{2}, 9, length(varargin{2}));
                this.weights = reshape(varargin{3}, 1, length(varargin{3}));
                this.n_components = length(this.weights);
            elseif (nargin == 5)
                this.n_components = varargin{4};
                this.means = reshape(varargin{1}, 3, this.n_components);
                this.covs = reshape(varargin{2}, 9, this.n_components);
                this.weights = reshape(varargin{3}, 1, this.n_components);
                this.support_size= varargin{5};
            else
                this.means = [];
                this.covs = [];
                this.weights = [];
                this.support_size = 0;
            end
        end
	function createMexObject(this)
	  this.objectHandle = gmm_mex('create_mex_object', ...
				      this.means, this.covs, ...
				      this.weights', this.support_size)
	end
        function score = aic(this, X)
            score = gmm_mex('aic', this.objectHandle(), X);
        end
        function score = bic(this, X)
            score = gmm_mex('bic', this.objectHandle(), X);
        end
        function getParameters(this)
        %wrapper = @(x) reshape(x, 1, 3, 3);
            [this.weights, this.means, this.covs, this.n_components, this.support_size] ...
                = gmm_mex('get_parameters', this.objectHandle());
            %this.covs = cell2mat(cellfun(wrapper, num2cell(covs,1), 'UniformOutput', false)');
            this.weights = this.weights';
        end
        function setSupportSize(this, n_support_size)
            gmm_mex('set_support_size', this.objectHandle, n_support_size);
            getParameters(this);
        end
        function setObjectHandle(this, oh)
            this.objectHandle = oh;
        end
        function oh = getObjectHandle(this)
            oh = this.objectHandle;
        end
        function poseUpdate(this, R, t)
            for i = 1:length(this.weights)
                cov = reshape(this.covs(:,i), 3, 3);
                cov_rotated = R*cov*R';
                this.covs(:, i) = reshape(cov_rotated, 9, 1);
                this.means(:, i) = R*this.means(:,i) + t;
            end
        end
        function transform(this, R, t)
            gmm_mex('transform', this.objectHandle, [R t; 0 0 0 1]);
            getParameters(this);
        end
        function isoplanarCovariances(this)
            for i = 1:length(this.weights)
                cov = reshape(this.covs(:,i), 3, 3);
                [U,S,V] = svd(cov);
                I = eye(3);
                I(3,3) = 0.001;
                new_cov = U*I*U';
                this.covs(:,i) = reshape(new_cov, 9, 1);
            end
        end
        function merge(this, that)
            this.means = [this.means, that.means];
            this.covs = [this.covs, that.covs];
            this.weights = [this.weights*this.support_size, that.weights* ...
                            that.support_size] ./ (this.support_size + that.support_size);
            this.support_size = this.support_size + that.support_size;
            %gmm_mex('merge', this.objectHandle, that.getObjectHandle());
            %getParameters(this);
        end
        function save(this, filepath)
            means = this.means';
            covs = this.covs';
            weights = this.weights';

            %%.20 is to make sure there are enough decimal points that the
            %%covariances are stable
            dlmwrite(filepath, [means, covs, weights], 'precision', '%.20f');
        end
       function load(this, filename)
            data = load(filename);
            if (size(data,2) ~= 13)
                error('input data should be Nx13');
            end
            this.n_components = size(data,1);
            this.means = transpose(data(:,1:3));
            this.covs = transpose(data(:,4:12));
            this.weights = transpose(data(:,13));
        end
       function loadMex(this, filename)
            data = load(filename);
            if (size(data,2) ~= 13)
                error('input data should be Nx13');
            end
            this.objectHandle = gmm_mex('load_from_file', filename);
            getParameters(this);
        end
        function score = dcs(this, that)
            score = gmm_mex('dcs', this.objectHandle, that.getObjectHandle());
        end
        function score = logLikelihood(this, X)
            score = gmm_mex('log_likelihood', this.objectHandle, X);
        end
        function score = weightedLogProbability(this, X)
            score = gmm_mex('weighted_log_probability', this.objectHandle, X);
        end
	function ps = posterior(this,X)
	  if size(X,1) == 3
	    X = transpose(X);
	  end

	  means = this.means;
	  covs = this.covs;
	  weights = this.weights;
	  ps = zeros(length(weights), size(X,1)); % K x S
	  lls = [];

	  for j = 1:size(X,1)
	    x = X(j, :);
	    ll = 0.0;
	    for i = 1:length(weights)
	      cov = reshape(covs(:,i), 3, 3);
	      prec = inv(cov);
	      mu = means(:,i);
	      w = weights(i);
	      ps(i,j) = w * ( det(cov)^(-1/2) / ( (2*pi)^(3/2) ) ) * exp( -0.5 * transpose(x' - mu) * prec * (x' - mu) );
	      ll = ll + ps(i,j);
	    end
	    lls(end+1) = ll;
	    ps(:,j) = ps(:,j) / ll;
	  end

	  ps = transpose(ps);
	end
	function score = posteriorMex(this, X)
	  score = gmm_mex('posterior', this.objectHandle, X);
	end
        % This is "fast" relative to the MATLAB's sample function. It wraps
        % the sample function inside the GMMNBase class
        function samples =  fastSample(this, num_samples)
            samples = gmm_mex('fast_sample', this.objectHandle(), num_samples);
        end
        function samples =  sample(this, num_samples)
            covs = [];
            for i=1:length(this.weights)
                B = reshape(this.covs(:,i), 3, 3);
		%[n,m]=size(A);
		%B=A'+A;
		%B(1:n+1:end)=diag(A);

		if (min(eig(B)) >= 0)
		elseif (min(eig(B)) >= -1e-14)
		  fprintf('Component %d has small negative eigenvalue...fixing\n', i);
		  [V,D] = eig(B);
		  B = V*max(D,0)/V;
		else
		  fprintf('Component %d not symmetric positive definite...min eigenvalue is %f\n', i, min(eig(B)));
		end

                covs(:,:,i) = B;
            end
            gmpdf = gmdistribution(this.means', covs, this.weights');
            samples = random(gmpdf, num_samples);
        end
        function that = deepCopy(this)
            means = this.means;
            covs = this.covs;
            weights = this.weights;
            n_components = this.n_components;
            that = GMM3(means,covs,weights,n_components);
        end
        function means = getMeans(this)
            means = this.means;
        end
        function covs = getCovs(this)
            covs = this.covs;
        end
        function weights = getWeights(this)
            weights = this.weights;
        end
	function precs = getPrecs(this)
	  precs = gmm_mex('get_precs', this.objectHandle);
	end
	function precs = setPrecs(this)
	  gmm_mex('set_precs', this.objectHandle);
	end
        function n_components = getNComponents(this)
            n_components = this.n_components;
        end
        function makeCovsIsoplanar(this)
            gmm_mex('make_covs_isoplanar', this.objectHandle);
            getParameters(this);
        end
        function plot(this, color)
        %PLOT_GMM Generate a plot of a GMM
        %   The GMM given by gmm is plotted with components display in the given
        %   color.
            count_failures = 0; % Failed to draw
	    nsigma = 3; % Number of sigmas to visualize
	    alpha = 0.3;

            for ix = 1:this.n_components
                cov = reshape(this.covs(:,ix), 3, 3);
                [U, L] = eig(cov);
                r = sqrt( diag(L) )*nsigma;
                mu = this.means(:,ix);
                [xc, yc, zc] = ellipsoid(0, 0, 0, r(1), r(2), r(3));

                % rotate data with orientation matrix U and center M
                a = kron(U(:,1),xc); b = kron(U(:,2),yc); c = kron(U(:,3),zc);
                data = a+b+c; n = size(data,2);
                x = data(1:n,:) + mu(1);
                y = data(n+1:2*n,:) + mu(2);
                z = data(2*n+1:end,:) + mu(3);

                C = repmat(reshape(color, 1, 1, 3), n, n, 1);
                try
                    sc = surf(x,y,z, C, 'FaceAlpha', alpha);
                    shading interp
                    hold on
                catch
                    count_failures = count_failures+1;
                    fprintf('failed to draw %d components\n', count_failures);
                end
            end
        end
    end
end
