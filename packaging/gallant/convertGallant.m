function convertGallant(directory, stimuliType, dryRun)
if ~exist('dryRun', 'var')
    dryRun = false;
end
if ~exist('stimuliType', 'var')
    stimuliType = 'NatRev';
end
if ~exist('directory', 'var')
    directory = 'V2Data';
end
fprintf('directory=%s, stimuliType=%s, dryRun=%b\n', directory, stimuliType, dryRun);
addpath(genpath([directory, '/functions']));
scriptDir = fileparts(mfilename('fullpath'));
addpath(genpath([scriptDir, '/lib']));
files = glob([directory, '/V2Data*/', stimuliType, '/*/*summary_file.mat']);
fprintf('Found %d summary files\n', numel(files));
cells = cell(0);
cellNumResponses = NaN(0);
for i = 1:length(files)
    file = files{i};
    [subDir, ~, ~] = fileparts(file);
    summaries = load(file);
    summaries = summaries.celldata;
    for summary = summaries'
        %% read and align stimuli and responses
%         stimuliFile = [subDir, '/', summary.stimfile];
%         try
%             stimuli = loadimfile(stimuliFile);
%         catch ME
%             fprintf('ERROR: could not load stimuli file %s\n', stimuliFile);
%             continue
%         end
%         stimuli = uint8(stimuli);
        responseFile = [subDir, '/', summary.respfile];
        response = respload(responseFile);
        response = nanmean(response, 2);
        % TODO: 600 natural images are cross-validation images
%         assert(length(response) == size(stimuli, 3));
        nonnan = ~isnan(response);
        fprintf('cell %s: keptFixation for %d/%d responses (%.2f%%)\n', ...
            summary.cellid, sum(nonnan), numel(response), sum(nonnan) / numel(response) * 100);
        cells{end + 1} = summary.cellid;
        cellNumResponses(end + 1) = sum(nonnan);
%         stimuli = stimuli(:, :, nonnan);
        response = response(nonnan);
        assert(~any(response < 0));
        continue;

        %% write stimuli, create data table
        stimuliPaths = writeStimuli(stimuli, stimuliFile, dryRun);
        [~, stimulusCategory] = fileparts(directory);
        stimulusCategory = repmat({stimulusCategory}, size(response));
        cellName = repmat({summary.cellid}, size(response));
        animal = repmat({summary.cellid(1)}, size(response));
        area = repmat({summary.area}, size(response));
        stimulusRepeats = repmat({summary.repcount}, size(response));
        data = table(stimuliPaths, response, stimulusCategory, cellName, area, animal, stimulusRepeats);
        
        %% write csv
        [responseFiledir, responseFilename, responseFileext] = fileparts(responseFile);
        dataFiledir = [responseFiledir, '/../data/'];
        csvPath = [dataFiledir, responseFilename, responseFileext, '.csv'];
        if ~dryRun
            if ~isfolder(dataFiledir)
                mkdir(dataFiledir);
            end
            writetable(data, csvPath);
        end
        fprintf('Wrote to %s\n', csvPath);
    end
end
cells = cells';
cellNumResponses = cellNumResponses';
data = table(cells, cellNumResponses);
writetable(data, [scriptDir, '/stats.csv']);
end

function stimuliPaths = writeStimuli(stimuli, stimuliFile, dryRun)
    [stimuliFile, basename, basenameExt] = fileparts(stimuliFile); 
    [stimuliDir, baseDir, ~] = fileparts(stimuliFile);
    stimuliDir = [stimuliDir, '/stimuli/'];
    if isfolder(stimuliDir)
        fprintf('directory %s exists already\n', stimuliDir);
    else
        mkdir(stimuliDir);
    end
    stimuliPaths = cell(size(stimuli, 3), 1);
    for stimulusNum = 1:size(stimuli, 3)
        image = stimuli(:, :, stimulusNum);
        stimuliPaths{stimulusNum} = [stimuliDir, '/', hashImage(image), '.jpg'];
        if ~dryRun
            % assert(~isfile(stimuliPaths{stimulusNum})); % some files are duplicates
            imwrite(image, stimuliPaths{stimulusNum});
        end
    end
end

function hashed = hashImage(image)
    hashed = DataHash(image);
end
