function [LLH] = model_llh(params, data, N, T)
p.rho_1 = params(1);
p.rho_2 = params(2);
p.phi_1 = params(3);
p.phi_2 = params(4);
p.beta  = params(5);
p.sigma_eps = params(6);
p.sigma_1 = params(7);
p.sigma_2 = params(8);

T = min(T, length(data));

data_logA=log(data(:,1));
data_B=data(:,2);
%%% Model-implied transition equations:
% state transition:

% long-run distribution:
rng(0)
lr_sim=5000;
x_dist =zeros(lr_sim+3,1);
dist_shocks=p.sigma_eps*randn(lr_sim+3,1);

for t=3: lr_sim+3
    x_dist(t)=p.rho_1*x_dist(t-1)+p.rho_2*x_dist(t-2)+p.phi_1*dist_shocks(t-1)+...
        p.phi_2*dist_shocks(t-2);
end

%%% Empirical log-likelihoods by particle filtering
% initialize particles 
particles = zeros(T, N,6);
llhs = zeros(T,1);

init_sample = randsample(lr_sim,N);
%initial set of states:
particles(1,:,1)=x_dist(init_sample+2); %X(t)
particles(1,:,2)=x_dist(init_sample+1); %X(t-1)
particles(1,:,3)=x_dist(init_sample);   %X(t-2)
particles(1,:,4)=dist_shocks(init_sample+2); %eps(t)
particles(1,:,5)=dist_shocks(init_sample+1); %eps(t-1)
particles(1,:,6)=dist_shocks(init_sample); %eps(t-2)

llhs(1) = log( mean( exp( ...
        log( normpdf(data_logA(1), particles(1,:,1), p.sigma_1) ) + ...
        log( normpdf(data_B(1), p.beta*particles(1,:,1) .^ 2, p.sigma_2) ) ...
        ) ) );

% predict, filter, update particles and collect the likelihood 

for t = 2:T
    %%% Prediction:
    particles(t,:,1)=p.rho_1*particles(t-1,:,1)+p.rho_2*particles(t-1,:,2)+...
        p.phi_1*particles(t-1,:,4)+p.phi_2*particles(t-1,:,5)+ p.sigma_eps*randn(1,N);
    particles(t,:,2)=particles(t-1,:,1);
    particles(t,:,3)=particles(t-1,:,2);
    particles(t,:,4)=p.sigma_eps*rand(1,N);
    particles(t,:,5)=particles(t-1,:,4);
    particles(t,:,6)=particles(t-1,:,5);
   
    %%% Filtering:
    llh = log( normpdf(data_logA(t), particles(t,:,1), p.sigma_1) ) + ...
        log( normpdf(data_B(t), p.beta*particles(t,:,1).^2, p.sigma_2) );
    lh = exp(llh);
    weights_s = exp( llh - log( sum(lh) ) );
    % store the log(mean likelihood)
    if sum(llh)==0
        weights_s(:)=1/length(weights_s);
    end
    
    llhs(t) = log(mean(lh));
    
    %%% Sampling:
    particles(t,:,1) = datasample(particles(t,:,1), N, 'Weights', weights_s);
    
end

LLH = sum(llhs);