function [F_observer_main_effect, ...
          p_value_main_effect, ...
          F_stage_observer_interaction, ...
          p_value_interaction] = run_network_adv_eval_anova_from_data_matrix_path_smaller_range(eval_models, ...
                                                                                  attack_name, ...
                                                                                  data_matrix_path, ...
                                                                                  num_bootstraps)
% Runs the non=parametric anova to compare the network predictions for different eps strengths
% Data matrix stored at data_matrix_path should be in the shape [num_seeds, num_eps_values, num_networks]
    if contains(attack_name,'l1')
        index_range = 4:7 
    elseif contains(attack_name,'l2')
        index_range = 3:6
    elseif contains(attack_name,'linf')
        index_range = 2:5
    end

    disp([newline 'Loading ' attack_name ' ||| ' data_matrix_path]) 
    load(data_matrix_path)

    disp(['Running EPS values [' num2str(eps_values(index_range)) '] for attack type ' attack_name])
    model_idx = arrayfun(@(t)(strmatch(t, networks, 'exact')), eval_models)
    anova_comparison_data_matrix = adversarial_eval_data_matrix(:,index_range,model_idx)

    % Stack the last dimension so we can run the between factor analysis
    disp('Using between factors')
    split_mat = num2cell(anova_comparison_data_matrix, [1 2]);
    data_matrix_between_factors = vertcat(split_mat{:})

    % Thi should create a vector specifying the factors
    single_net_factor = ones(size(anova_comparison_data_matrix(:,:,1), 1),1)
    network_between_factors_init = single_net_factor * (1:size(anova_comparison_data_matrix,3))
    network_between_factors_cell = num2cell(network_between_factors_init, [1])
    network_between_factors = vertcat(network_between_factors_cell{:})

    within_factor_names = {'eps'};
    between_factor_names = {'network'};
    tbl = simple_mixed_anova_partialeta(data_matrix_between_factors, network_between_factors, within_factor_names, between_factor_names);

    [p_value_main_effect, p_value_interaction] = run_permutation_significance_test_model_adv_eval(...
        tbl, data_matrix_between_factors, ...
        network_between_factors, within_factor_names, ...
        between_factor_names, num_bootstraps ...
        );

    disp([attack_name ' Full ANOVA'])
    disp(tbl)
    F_observer_main_effect = tbl{'network','F'};
    disp([attack_name ' F(observer) main effect: ' num2str(F_observer_main_effect)])
    disp([attack_name ' p-value main effect: ' num2str(p_value_main_effect)])
    F_stage_observer_interaction = tbl{'network:eps','F'};
    disp([attack_name ' F(eps, observer) interaction: ' num2str(tbl{'network:eps','F'})])
    disp([attack_name ' p-value interaction: ' num2str(p_value_interaction)])
end
