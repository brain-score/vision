function [p_value_main_effect] = run_permutation_significance_test_layer_effect_human(...
    true_values_statistics, data_matrix, ...
    within_factor_names, ...
    num_bootstraps ...
    )

% Computes p-values for a main effect of layer by shuffling the layer labels. 

[num_participants, num_layers] = size(data_matrix);

% permute the layers (independent for each participant). 
all_bootstraps_F_main_effect = nan(1,num_bootstraps);

rowIndex = repmat((1:num_participants)',[1 num_layers]);

parfor bootstrap_idx=1:num_bootstraps
    % Randomly permute the layers as well, independent for each
    % participant. 
    [~,randomizedColIndex] = sort(rand(num_participants,num_layers),2);
    newLinearIndex = sub2ind([num_participants,num_layers],rowIndex,randomizedColIndex);
    randomized_data_matrix = data_matrix(newLinearIndex);
    bootstrap_tbl = simple_mixed_anova_partialeta(randomized_data_matrix, ...
        [], within_factor_names);
    all_bootstraps_F_main_effect(bootstrap_idx) = bootstrap_tbl{['(Intercept):', char(within_factor_names(1))],'F'};
end

p_value_main_effect = (sum(all_bootstraps_F_main_effect >= true_values_statistics{['(Intercept):', char(within_factor_names(1))],'F'}))/(num_bootstraps);
if p_value_main_effect == 0
    p_value_main_effect = 1/num_bootstraps;
end

end
