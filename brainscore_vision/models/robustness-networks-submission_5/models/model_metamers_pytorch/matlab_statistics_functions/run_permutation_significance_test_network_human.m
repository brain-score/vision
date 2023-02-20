function [p_value_main_effect, p_value_interaction] = run_permutation_significance_test_network_human(...
    true_values_statistics, data_matrix_between_factors, ...
    network_human_between_factors, within_factor_names, ...
    between_factor_names, num_bootstraps ...
    )

% [p_value_main_effect, p_value_interaction] = run_permutation_significance_test_network_human(...
%     true_values_statistics, data_matrix_between_factors, ...
%     network_human_between_factors, within_factor_names, ...
%     between_factor_names, num_bootstraps ...
%     )
% 
% Computes p-values for an interaction between layer and observer type, and a p-value for the 
% main effect of observer by shuffling the between factor labels (main efffect) and the 
% values for each layer (for interaction). This makes a null distribution for the F-statistic
% and the p-value is computed by comparing the true F with the null values. 

[num_participants, num_layers] = size(data_matrix_between_factors);

% First bootstrap the interaction -- permute both human-network assignment
% and the layers (independent for each participant). 
all_bootstraps_F_interaction = nan(1,num_bootstraps);
length_factors = length(network_human_between_factors);

rowIndex = repmat((1:num_participants)',[1 num_layers]);

parfor bootstrap_idx=1:num_bootstraps
    % randomly permute the human vs. network assignment
    randomized_network_human_factor = network_human_between_factors(randperm(length_factors));
    % Randomly permute the layers as well, independent for each
    % participant. 
    [~,randomizedColIndex] = sort(rand(num_participants,num_layers),2);
    newLinearIndex = sub2ind([num_participants,num_layers],rowIndex,randomizedColIndex);
    randomized_data_matrix_between_factors = data_matrix_between_factors(newLinearIndex);
    bootstrap_tbl = simple_mixed_anova_partialeta(randomized_data_matrix_between_factors, ...
        randomized_network_human_factor, within_factor_names, between_factor_names);
    all_bootstraps_F_interaction(bootstrap_idx) = bootstrap_tbl{'network_or_human:layer','F'};
end

p_value_interaction = (sum(all_bootstraps_F_interaction >= true_values_statistics{'network_or_human:layer','F'}))/(num_bootstraps);
if p_value_interaction == 0
    p_value_interaction = 1/num_bootstraps;
end

% Now bootstrap the main effect of network/human -- just permute the human 
% and network labels (also average across layer, which saves computation)
all_bootstraps_F_main_effect = nan(1,num_bootstraps);

parfor bootstrap_idx=1:num_bootstraps
    % randomly permute the human vs. network assignment
    randomized_network_human_factor = network_human_between_factors(randperm(length_factors));
%     bootstrap_tbl = simple_mixed_anova_partialeta(data_matrix_between_factors, ...
%         randomized_network_human_factor, within_factor_names, between_factor_names);
% This can be used instead to save computation for the main effect.  
    bootstrap_tbl = simple_mixed_anova_partialeta(repmat(mean(data_matrix_between_factors,2),[1,2]), ...
        randomized_network_human_factor, within_factor_names, between_factor_names);
    all_bootstraps_F_main_effect(bootstrap_idx) = bootstrap_tbl{'network_or_human', 'F'};
end

p_value_main_effect = (sum(all_bootstraps_F_main_effect >= true_values_statistics{'network_or_human','F'}))/(num_bootstraps);
if p_value_main_effect == 0
    p_value_main_effect = 1/num_bootstraps;
end

end
