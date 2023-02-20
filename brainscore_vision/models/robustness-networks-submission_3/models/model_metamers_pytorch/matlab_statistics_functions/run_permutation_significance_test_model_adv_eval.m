function [p_value_main_effect, p_value_interaction] = run_permutation_significance_test_network_human(...
    true_values_statistics, data_matrix_between_factors, ...
    network_observer_between_factors, within_factor_names, ...
    between_factor_names, num_bootstraps ...
    )

% [p_value_main_effect, p_value_interaction] = run_permutation_significance_test_network_human(...
%     true_values_statistics, data_matrix_between_factors, ...
%     network_observer_between_factors, within_factor_names, ...
%     between_factor_names, num_bootstraps ...
%     )
% 
% Computes p-values for an interaction between EPS value and obverser type, and a p-value for the 
% main effect of observer by shuffling between factor labels (main effect) and the values for each 
% eps value (for interaction). This makes a null distribution for the F-stat and the p-valu
% is computed by comparing the true F with the null values

[num_seeds, num_eps_vals] = size(data_matrix_between_factors);

% First bootstrap the interaction -- permute both network assignment
% and the eps. 
all_bootstraps_F_interaction = nan(1,num_bootstraps);
length_factors = length(network_observer_between_factors);

rowIndex = repmat((1:num_seeds)',[1 num_eps_vals]);

parfor bootstrap_idx=1:num_bootstraps
    randomized_network_observer_factor = network_observer_between_factors(randperm(length_factors));
    % Randomly permute the layers as well, independent for each
    % seed. 
    [~,randomizedColIndex] = sort(rand(num_seeds,num_eps_vals),2);
    newLinearIndex = sub2ind([num_seeds,num_eps_vals],rowIndex,randomizedColIndex);
    randomized_data_matrix_between_factors = data_matrix_between_factors(newLinearIndex);
    bootstrap_tbl = simple_mixed_anova_partialeta(randomized_data_matrix_between_factors, ...
        randomized_network_observer_factor, within_factor_names, between_factor_names);
    all_bootstraps_F_interaction(bootstrap_idx) = bootstrap_tbl{'network:eps','F'};
end

p_value_interaction = (sum(all_bootstraps_F_interaction > true_values_statistics{'network:eps','F'}))/(num_bootstraps);
if p_value_interaction == 0
    p_value_interaction = 1/num_bootstraps;
end

% Now bootstrap the main effect of network
all_bootstraps_F_main_effect = nan(1,num_bootstraps);

parfor bootstrap_idx=1:num_bootstraps
    randomized_network_observer_factor = network_observer_between_factors(randperm(length_factors));
    bootstrap_tbl = simple_mixed_anova_partialeta(repmat(mean(data_matrix_between_factors,2),[1,2]), ...
        randomized_network_observer_factor, within_factor_names, between_factor_names);
    all_bootstraps_F_main_effect(bootstrap_idx) = bootstrap_tbl{'network', 'F'};
end

p_value_main_effect = (sum(all_bootstraps_F_main_effect > true_values_statistics{'network','F'}))/(num_bootstraps);
if p_value_main_effect == 0
    p_value_main_effect = 1/num_bootstraps;
end

end
