% function R = compute_corr_manual(X)
%     mu_X = mean(X);
%     sigma_X = std(X);
%     X_norm = (X - mu_X) ./ sigma_X;
%     R = (X_norm' * X_norm) / (size(X, 1) - 1);
% end

function R = compute_corr_manual(X)
    mu_X = mean(X); % Mean of each column
    centered_X = X - mu_X; % Center the data
    n = size(X, 1); % Number of observations
    sigma_X = std(centered_X); % Standard deviation
    X_norm = centered_X ./ sigma_X; % Normalizing the data
    R = (X_norm' * X_norm) / (n - 1); % Computing the correlation matrix
end