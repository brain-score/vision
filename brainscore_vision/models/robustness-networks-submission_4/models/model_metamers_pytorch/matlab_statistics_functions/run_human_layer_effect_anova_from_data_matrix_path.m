function [F_observer_main_effect, p_value_main_effect, F_stage_observer_interaction, p_value_interaction] = run_human_layer_effect_anova_from_data_matrix_path(model_name, data_matrix_path, num_bootstraps)
% Runs the non=parametric anova to compare the network responses to the human responses.
% Data matrix stored at data_matrix_path should be in the shape [num_participants, num_layers, human/network]
    disp([newline 'Loading ' model_name ' ||| ' data_matrix_path])
    load(data_matrix_path)

    data_matrix = participant_data_matrix(:,:,1);

    % Remove the final classification layer from the analysis. 
    if contains(model_name, 'hmax', 'IgnoreCase', true)
        disp('Removing Human Recognition of Classifier Metamers for HMAX ANOVA')
        data_matrix = data_matrix(:,1:end-1,:);
    elseif contains(model_name, 'spectemp', 'IgnoreCase', true)
        disp('Removing Human Recognition of Classifier Metamers for SPECTEMP ANOVA')
        data_matrix = data_matrix(:,1:end-1,:);
    end

    disp(data_matrix)

    within_factor_names = {'stage'};
    tbl = simple_mixed_anova_partialeta(data_matrix, [], within_factor_names);

    [p_value_main_effect] = run_permutation_significance_test_layer_effect_human(...
        tbl, data_matrix, ...
        within_factor_names, ...
        num_bootstraps ...
        );

    disp([model_name ' Full ANOVA'])
    disp(tbl)
    F_observer_main_effect = tbl{['(Intercept):', char(within_factor_names(1))],'F'};
    disp([model_name ' F(observer) main effect: ' num2str(F_observer_main_effect)])
    disp([model_name ' p-value main effect: ' num2str(p_value_main_effect)])
end
