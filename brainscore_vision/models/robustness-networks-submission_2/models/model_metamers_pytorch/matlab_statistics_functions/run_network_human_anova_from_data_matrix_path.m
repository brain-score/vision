function [F_observer_main_effect, p_value_main_effect, F_stage_observer_interaction, p_value_interaction] = run_network_human_anova_from_data_matrix_path(model_name, data_matrix_path, num_bootstraps)
% Runs the non=parametric anova to compare the network responses to the human responses.
% Data matrix stored at data_matrix_path should be in the shape [num_participants, num_layers, human/network]
    disp(['Loading ' model_name ' ||| ' data_matrix_path])
    load(data_matrix_path)

    disp('Using between factors')
    data_matrix_between_factors = cat(1,participant_data_matrix(:,:,1), participant_data_matrix(:,:,2));
    network_human_between_factors = cat(1,logical(zeros(size(participant_data_matrix(:,:,1), 1),1)), logical(ones(size(participant_data_matrix(:,:,2), 1),1)));
    within_factor_names = {'layer'};
    between_factor_names = {'network_or_human'};
    tbl = simple_mixed_anova_partialeta(data_matrix_between_factors, network_human_between_factors, within_factor_names, between_factor_names);

    [p_value_main_effect, p_value_interaction] = run_permutation_significance_test_network_human(...
        tbl, data_matrix_between_factors, ...
        network_human_between_factors, within_factor_names, ...
        between_factor_names, num_bootstraps ...
        );

    disp([model_name ' Full ANOVA'])
    disp(tbl)
    F_observer_main_effect = tbl{'network_or_human','F'};
    disp([model_name ' F(observer) main effect: ' num2str(F_observer_main_effect)])
    disp([model_name ' p-value main effect: ' num2str(p_value_main_effect)])
    F_stage_observer_interaction = tbl{'network_or_human:layer','F'};
    disp([model_name ' F(stage, observer) interaction: ' num2str(tbl{'network_or_human:layer','F'})])
    disp([model_name ' p-value interaction: ' num2str(p_value_interaction)])
end
