function [tbl,rm] = simple_mixed_anova_partialeta(datamat, varargin)

% tbl = simple_mixed_anova(datamat, between_factors, within_factor_names,
% between_factor_names)
%
% Repeated-measures or mixed ANOVA with any number of factors.
%
% Function built on top of existing MATLAB functions in an attempt to
% simplify user inputs and manipulations while still being able to perform
% all common ANOVA designs.
%
% DATAMAT is a numerical matrix containing the values associated with each
% level of each within-subject factor in each subject (responses). The
% subjects should always be the first dimension of the matrix. Each
% within-subject factor is another dimension.
%
% BETWEEN_FACTORS is a numerical matrix where each row represents a subject
% and each column represents a between-subjects factor. A given value
% represents the level of the column's factor associated with the row's
% subject. Optional.
%
% WITHIN_FACTOR_NAMES is a cell array of strings indicating the name of
% each within-subject factor (one for each dimension of the datamat
% variable except the first, in order). Optional.
%
% BETWEEN_FACTOR_NAMES is a cell array of strings indicating the name of
% each between-subjects factor (one for each column of the between_factors
% matrix, in order). These factors are assumed to be categorical (groups).
% Optional.
%
% TABLE is a table indicating the F statistics, p-values and other
% statistics associated with each term of the model. The "intercept" terms
% can be ignored (e.g. "(Intercept):WS01" indicates the main effect of
% WS01).
%
% RM is a structure with the repeated measures model parameters and
% statistics.
%
% Does not support covariates or partial models (without all interactions)
% for now.
%
%
% EXAMPLE
%
% A design with 24 subjects and 2 within-subject factors, the first one
% having 3 levels (time: pre-test, 1st post-test, 2nd post-test) and the
% second one 4 levels (experimental condition: A, B, C, D). The subjects
% are grouped in 4 groups: 2 variables with 2 levels each (gender: male or
% female; age group: young or old).
%
% The input datamat should be a 24 x 3 x 4 matrix with each row
% corresponding to a subject. So the element datamat(1,1,1) will correspond
% to the response of the subject #1 in the pre-test in experimental
% condition A, the element datamat(2,3,2) will correspond to the response
% of the subject #2 in the 2nd post-test in experimental condition B, and
% so on.
%
% The input between_factors will be a 24 x 2 matrix with each row
% corresponding to a subject and each column to a between-subjects factor
% (gender and age). Each column will be filled with 1s and 2s, or
% 0s and 1s, or other numbers, indicating the gender/age group of the
% respective subject.
%
% tbl = simple_mixed_anova(datamat, between_factors, {'Time', 'Exp_cond'},
% {'Gender', 'Age_group'})
%
% Copyright 2017, Laurent Caplette
% https://www.researchgate.net/profile/Laurent_Caplette

% Modified, May 2020 by M. McPherson to add partial eta squared for 1-3
% repeated measures and 1 within subject effect.

% Check if correct number of inputs
narginchk(1,4)

% Assign inputs to variables; if none, will be empty array
between_factors = [];
within_factor_names = [];
between_factor_names = [];
if nargin>1
    between_factors = varargin{1};
    if nargin>2
        within_factor_names = varargin{2};
        if nargin>3
            between_factor_names = varargin{3};
        end
    end
end

% Determine numbers of variables and measures
nWithin = ndims(datamat)-1;
nBetween = size(between_factors,2);
nVars = size(datamat);
nVars = nVars(2:end); % don't use the nb of subjects on the first dim
nMeas = prod(nVars);

% Check if dimensions of matrices are ok
if size(datamat,1)<2
    error('There must be more than one subject.')
end
if ~isempty(between_factors)
    if size(between_factors,1)~=size(datamat,1)
        error('Both input matrices must have the same nb of subjects.')
    end
end

% Check if there is more than one unique value
if length(unique(datamat))<2
    error('The data matrix must contain more than one unique value.')
end
for ii = 1:size(between_factors,2)
    if length(unique(between_factors(:,ii)))<2
        error('Each between-subjects factor must contain more than one unique value.')
    end
end

% Error if more variable names than variables as input
if length(between_factor_names)>nBetween
    error('Too many between-subject factor names or not enough between-subject variables as input.')
end
if length(within_factor_names)>nWithin
    error('Too many within-subject factor names or not enough within-subject variables as input.')
end

% Check validity of variable names
for ii = 1:length(between_factor_names)
    if ~isvarname(between_factor_names{ii})
        error('Variable names must be continuous strings starting with a letter and without symbols.')
    end
end
for ii = 1:length(within_factor_names)
    if ~isvarname(within_factor_names{ii})
        error('Variable names must be continuous strings starting with a letter and without symbols.')
    end
end

% Assign variable names if not enough or empty
if length(between_factor_names)<nBetween
    nMissing = nBetween - length(between_factor_names);
    BS = repmat('BS', [nMissing 1]); % list of 'BS'
    missing_factor_names = cellstr([BS num2str([1:nMissing]', '%02.0f')]);
    between_factor_names = [between_factor_names missing_factor_names];
end
if length(within_factor_names)<nWithin
    nMissing = nWithin - length(within_factor_names);
    WS = repmat('WS', [nMissing 1]); % list of 'WS'
    missing_factor_names = cellstr([WS num2str([1:nMissing]', '%02.0f')]);
    within_factor_names = [within_factor_names missing_factor_names];
end

% Create table detailing within-subject design
withinVarLevels = fullfact(nVars); % all level combinations
within_table = array2table(withinVarLevels, 'VariableNames', within_factor_names);
for ii = 1:nWithin % ensure that each within-subject factor is categorical (levels==discrete)
    evalc(sprintf('within_table.%s = categorical(within_table.%s)', within_factor_names{ii}, within_factor_names{ii}));
end

% Vectorize all dimensions after first one of the data matrix
y = datamat(:,:);

% Create data table
yList = repmat('Y', [nMeas 1]); % list of 'Y'
numList = num2str([1:nMeas]', '%03.0f'); % support up to 999 measures
measureNames = cellstr([yList numList]); % create names for every measure
for ii = 1:nBetween % add between-subject factors
    measureNames{nMeas+ii} = between_factor_names{ii};
end
total_table = array2table([y between_factors],'VariableNames', measureNames);
for ii = 1:nBetween % ensure that each between-subject factor is categorical (levels/groups==discrete)
    evalc(sprintf('total_table.%s = categorical(total_table.%s)', between_factor_names{ii}, between_factor_names{ii}));
end

% Create between-subjects model using Wilkinson notation
betweenModel = '';
for ii = 1:nBetween
    betweenModel = [betweenModel,measureNames{nMeas+ii},'*'];
end
betweenModel = betweenModel(1:end-1); % remove last star
if isempty(betweenModel)
    betweenModel = '1'; % if no between-subjects factor, put constant term (usually implicit)
end

% Create within-subject model using Wilkinson notation
withinModel = '';
for ii = 1:nWithin
    withinModel = [withinModel,within_factor_names{ii},'*']; % stars for full model (all interactions)
end
withinModel = withinModel(1:end-1); % remove last star

% Fit repeated measures model
rm = fitrm(total_table, sprintf('%s-%s~%s', measureNames{1}, measureNames{nMeas}, betweenModel),...
    'WithinDesign', within_table);

% Run ANOVA
tbl = ranova(rm, 'WithinModel', withinModel);
%tbl_Properties_VariableNames =
partial_etaSquared = NaN(size(tbl,1), 1);


for i = 1:nWithin
    Sum_Squares_Col = find(strcmp('SumSq', tbl.Properties.VariableNames)==1);
    
    % Main Effects
    
    ind_Intercept = find(strcmp(sprintf('(Intercept):%s',within_factor_names{i}), tbl.Properties.RowNames)==1);
    ind_Error = find(strcmp(sprintf('Error(%s)',within_factor_names{i}), tbl.Properties.RowNames)==1);
    partial_etaSquared(ind_Intercept) = tbl{ind_Intercept,Sum_Squares_Col}/(tbl{ind_Intercept,Sum_Squares_Col}+tbl{ind_Error,Sum_Squares_Col});
    
    %Interactions with between Subject Effects, if present
    
    ind_Intercept = find(strcmp(sprintf('(Intercept):%s',within_factor_names{i}), tbl.Properties.RowNames)==1);
    ind_Error = find(strcmp(sprintf('Error(%s)',within_factor_names{i}), tbl.Properties.RowNames)==1);
    partial_etaSquared(ind_Intercept) = tbl{ind_Intercept,Sum_Squares_Col}/(tbl{ind_Intercept,Sum_Squares_Col}+tbl{ind_Error,Sum_Squares_Col});
    
    %Interaction between between Subject Effects
    
end
if nWithin==2
    ind_Intercept = find(strcmp(sprintf('(Intercept):%s:%s',within_factor_names{1},within_factor_names{2}), tbl.Properties.RowNames)==1);
    ind_Error = find(strcmp(sprintf('Error(%s:%s)',within_factor_names{1},within_factor_names{2}), tbl.Properties.RowNames)==1);
    partial_etaSquared(ind_Intercept) = tbl{ind_Intercept,Sum_Squares_Col}/(tbl{ind_Intercept,Sum_Squares_Col}+tbl{ind_Error,Sum_Squares_Col});
end

if nBetween ==1  % Between Subject Effects
    %ind_TotalIntercept = find(strcmp('(Intercept)', tbl.Properties.RowNames)==1);
    ind_BetweenSubjectIntercept = find(strcmp(between_factor_names, tbl.Properties.RowNames)==1);
    ind_Error = find(strcmp(sprintf('Error'), tbl.Properties.RowNames)==1);
    partial_etaSquared(ind_BetweenSubjectIntercept) = tbl{ind_BetweenSubjectIntercept,Sum_Squares_Col}/(tbl{ind_BetweenSubjectIntercept,Sum_Squares_Col}+tbl{ind_Error,Sum_Squares_Col});
    
    for i = 1:nWithin % Interactions with Between Subject Effects
        Sum_Squares_Col = find(strcmp('SumSq', tbl.Properties.VariableNames)==1);
        
        % Main Effects
        
        ind_Intercept = find(strcmp(sprintf('%s:%s',char(between_factor_names),within_factor_names{i}), tbl.Properties.RowNames)==1);
        ind_Error = find(strcmp(sprintf('Error(%s)',within_factor_names{i}), tbl.Properties.RowNames)==1);
        partial_etaSquared(ind_Intercept) = tbl{ind_Intercept,Sum_Squares_Col}/(tbl{ind_Intercept,Sum_Squares_Col}+tbl{ind_Error,Sum_Squares_Col});
        
        %Interactions with between Subject Effects, if present
        
        ind_Intercept = find(strcmp(sprintf('(%s:%s',char(between_factor_names),within_factor_names{i}), tbl.Properties.RowNames)==1);
        ind_Error = find(strcmp(sprintf('Error(%s)',within_factor_names{i}), tbl.Properties.RowNames)==1);
        partial_etaSquared(ind_Intercept) = tbl{ind_Intercept,Sum_Squares_Col}/(tbl{ind_Intercept,Sum_Squares_Col}+tbl{ind_Error,Sum_Squares_Col});
        
        %Interaction between between Subject Effects
        
    end
    if nWithin==2
        ind_Intercept = find(strcmp(sprintf('%s:%s:%s',char(between_factor_names),within_factor_names{1},within_factor_names{2}), tbl.Properties.RowNames)==1);
        ind_Error = find(strcmp(sprintf('Error(%s:%s)',within_factor_names{1},within_factor_names{2}), tbl.Properties.RowNames)==1);
        partial_etaSquared(ind_Intercept) = tbl{ind_Intercept,Sum_Squares_Col}/(tbl{ind_Intercept,Sum_Squares_Col}+tbl{ind_Error,Sum_Squares_Col});
    end
    
end



tbl = addvars(tbl,partial_etaSquared,'After','pValueLB');
end

