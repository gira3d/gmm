classdef GMM4 < handle
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
        function this = GMM4(varargin)
	  if nargin > 0
	    filepath = varargin{1};
	    this.loadData(filepath);
	  else
	  end
	end
	function fit(this, X, n_clusters)
          [this.weights, this.means, this.covs, this.objectHandle] = gmm4_mex('new', X, n_clusters);
          this.n_components = n_clusters;
          this.support_size = size(X,2);
          this.weights = this.weights';
	end
	function [m,v] = regress(this, X)
	  [m_j, s2_j] = this.conditionalMeanVar(X);
	  w_j = this.conditionalWeights(X);

	  m = transpose(sum(transpose(w_j .* m_j)));

	  term1 = w_j .* (m_j .* m_j + s2_j);
	  term2 = (w_j .* m_j);

	  v = transpose(sum(transpose(term1))) - transpose(sum(transpose(term2))).^2;
	end
	function [m,v] = regressMex(this, X)
	  [m, v] = gmm4_mex('regress', this.objectHandle, X);
	end
	function w = conditionalWeights(this, X)
	  gmm3 = this.get3DGMM();
	  w = gmm3.posterior(X);
	end
	function w = conditionalWeightsMex(this, X)
	  w = gmm4_mex('conditional_weights', this.objectHandle, X);
	end
	function [ms,vs] = conditionalMeanVar(this, X)

	  means = this.means;
	  covs = this.covs;
	  weights = this.weights;

	  ms = zeros(length(X), length(weights)); % S x K
	  vs = zeros(length(X), length(weights)); % S x K

	  for i = 1:length(X)
	    x = X(:,i);

	    for j = 1:length(weights)
	      cov = reshape(covs(:,j), 4, 4);

	      EjX = cov(1:3,1:3);
	      EjYX = cov(4,1:3);
	      EjXY = cov(1:3,4);
	      EjYY = cov(4,4);

	      PjX = inv(EjX);

	      mu = means(:,j);
	      mujY = mu(4);
	      mujX = mu(1:3);

	      w = weights(j);

	      ms(i,j) = mujY + EjYX * PjX * (x-mujX);
	      vs(i,j) = EjYY - EjYX * PjX * EjXY;
	    end
	  end
	end
	function [m,s2] = conditionalMeanVarMex(this, X)
	  [m, s2] = gmm4_mex('conditional_meanvar', this.objectHandle, X);
	end
	function gmm = get3DGMM(this)
	  gmm = GMM3();
	  gmm.weights = this.weights;
	  gmm.means = this.means(1:3,:);
	  gmm.covs = [];

	  for i = 1:length(gmm.weights)
	    cov = this.covs(:,i);
	    cov = reshape(cov, [4,4]);
	    cov = cov(1:3, 1:3);
	    cov = reshape(cov, [9,1]);
	    gmm.covs(:,i) = cov;
	  end
	end
	function gmm3 = get3DGMMMex(this)
	  handle = gmm4_mex('get3dgmm', this.objectHandle);
	  gmm3 = GMM3();
	  gmm3.setObjectHandle(handle);
	  gmm3.getParameters();
	end
        function save(this, filepath)
            means = this.means';
            covs = this.covs';
            weights = this.weights';

            %%.20 is to make sure there are enough decimal points that the
            %%covariances are stable
            dlmwrite(filepath, [means, covs, weights], 'precision', '%.20f');
        end
       function loadData(this, filename)
            data = load(filename);
            if (size(data,2) ~= 21)
                error('input data should be Nx21');
            end
	    this.means = transpose(data(:,1:4));
	    this.covs = transpose(data(:, 5:20));
	    this.weights = transpose(data(:,21));
        end
       function loadDataMex(this, filename)
	 [this.objectHandle] = gmm4_mex('load_from_file', filename);
	 [this.weights, this.means, this.covs, this.n_components, this.support_size] = gmm4_mex('get_parameters', this.objectHandle);
        end
        function transform(this, R, t)
	  if (size(R,1) == [3 3])
	    R2 = eye(4);
	    R2(1:3, 1:3) = R;
	    R = R2;
	  end

	  if (size(t) == [3 1])
	    t2 = zeros(4,1);
	    t2(1:3) = t;
	    t = t2;
	  end

          for i = 1:length(this.weights)
            cov = reshape(this.covs(:,i), 4, 4);
            cov = R*cov*R';
            this.covs(:, i) = reshape(cov, 16, 1);
            this.means(:, i) = R*this.means(:,i) + t;
          end
        end
        function [samples, pcld] =  sample(this, num_samples, weight)
            covs = [];
            for i=1:length(this.weights)
                B = reshape(this.covs(:,i), 4, 4);

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
	    pcld = pointCloud(samples(:,1:3));
	    pcld.Color = uint8([samples(:, 4), samples(:, 4), samples(:, 4)] * weight);
	end
    end
end
