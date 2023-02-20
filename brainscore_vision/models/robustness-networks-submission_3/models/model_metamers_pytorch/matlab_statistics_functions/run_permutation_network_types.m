function [p_value_interaction, p_value_main_effect_model] = run_permutation_network_types(...
    true_values_statistics, participant_data_matrix, ...
    within_factor_names, ...
    num_bootstraps ...
    )

% Test for a main effect between networks (factor 2) 
% To obtain null, first average across layers (to save computation), then
% permute the network labels (do not shuffle layers)  

% This bootstraps for the main effect of model
all_bootstraps_F_main_effect_model = nan(1,num_bootstraps);
[num_participants, ~, num_models] = size(participant_data_matrix);
   
% average across layers
reshaped_participant_data_matrix = squeeze(mean(participant_data_matrix,2));
rowIndex = repmat((1:num_participants)',[1 num_models]);
reshaped_participant_data_matrix

parfor bootstrap_idx=1:num_bootstraps
    % Randomly permute the models, independent for each participant
    [~,randomizedColIndex] = sort(rand(num_participants,num_models),2);
    newLinearIndex = sub2ind([num_participants,num_models],rowIndex,randomizedColIndex);
    randomized_data_matrix = reshaped_participant_data_matrix(newLinearIndex);
    
    bootstrap_tbl = simple_mixed_anova_partialeta(randomized_data_matrix, ...
        [], within_factor_names(2));
    all_bootstraps_F_main_effect_model(bootstrap_idx) = bootstrap_tbl{'(Intercept):model_type','F'};
end

p_value_main_effect_model = (sum(all_bootstraps_F_main_effect_model > true_values_statistics{'(Intercept):model_type','F'}))/(num_bootstraps);
if p_value_main_effect_model == 0
    p_value_main_effect_model = 1/num_bootstraps;
end

% Test for an interaction between layer (factor 1) and network (factor 2)
% To obtain null, permute the layer order and permute the network
% assignment. 

% This bootstraps for the interaction between stage and model
all_bootstraps_F_interaction = nan(1,num_bootstraps);
[num_participants, num_layers, num_models] = size(participant_data_matrix);

% flatten the layers and models, so that it is easy for us to shuffle
% across them. 
reshaped_participant_data_matrix = reshape(participant_data_matrix, num_participants, num_layers*num_models);
rowIndex = repmat((1:num_participants)',[1 num_layers*num_models]);

parfor bootstrap_idx=1:num_bootstraps
    % Randomly permute the layers and models, independent for each participant
    [~,randomizedColIndex] = sort(rand(num_participants,num_layers*num_models),2);
    newLinearIndex = sub2ind([num_participants,num_layers*num_models],rowIndex,randomizedColIndex);
    reshaped_randomized_data_matrix = reshaped_participant_data_matrix(newLinearIndex);
    randomized_data_matrix = reshape(reshaped_randomized_data_matrix, num_participants, num_layers, num_models);
    
    bootstrap_tbl = simple_mixed_anova_partialeta(randomized_data_matrix, ...
        [], within_factor_names);
    all_bootstraps_F_interaction(bootstrap_idx) = bootstrap_tbl{'(Intercept):layer:model_type','F'};
end

p_value_interaction = (sum(all_bootstraps_F_interaction > true_values_statistics{'(Intercept):layer:model_type','F'}))/(num_bootstraps);
if p_value_interaction == 0
    p_value_interaction = 1/num_bootstraps;
end

end

