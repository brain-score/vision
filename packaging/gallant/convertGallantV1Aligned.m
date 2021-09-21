function convertGallantV1Aligned(directory, dryRun)
if ~exist('dryRun', 'var')
    dryRun = false;
end
if ~exist('directory', 'var')
    directory = 'v1_nis_data';
end
stimuliDir = [directory, '/stimuli'];
addpath(genpath([directory, '/../functions']));
scriptDir = fileparts(mfilename('fullpath'));
addpath(genpath([scriptDir, '/lib']));

cellinfo();

for i = 1:length(celldata)
    stimuli = loadimfile([directory, '/', celldata(i).fullstimfile]);
    responseData = load([directory, '/', celldata(i).datafile]);
    response = responseData.resp;
    
    assert(length(response) == size(stimuli, 3));
    keptFixation = ~ismember(response, -1) & ~isnan(response);
    response = response(keptFixation);
    stimuli = stimuli(:, :, keptFixation);

    %% write stimuli, create data table
    fprintf('Writing stimuli to %s\n', stimuliDir);
    stimuliPaths = writeStimuli(stimuli, stimuliDir, dryRun);
    cellName = repmat({responseData.cellid}, size(response));
    area = repmat({'V1'}, size(response));
    data = table(stimuliPaths, response, cellName, area);

    %% write csv
    csvPath = [directory, '/', responseData.cellid, '.csv'];
    if ~dryRun
        writetable(data, csvPath);
    end
    fprintf('Wrote csv to %s\n', csvPath);
end
end

function stimuliPaths = writeStimuli(stimuli, stimuliDir, dryRun)
    if ~isfolder(stimuliDir)
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
