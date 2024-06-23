%% collect all scans and save
clear all

demos = readtable('tabular/demographics_short.csv');
drugs = readtable('tabular/cahalan_plus_drugs.csv');

sids = unique(drugs.subject); % find subjects we have cahalan for
demos = demos(ismember(demos.subject, sids) & strcmp(demos.arm, 'standard') & ~isnan(demos.mri_restingstate_age), :); % filter demos table to show only subjects we have chalan for

%% match rows across tables based on visit code & get mean FD

mrd = readtable('tabular/MRD.csv'); % mean FD values

% Define the mapping between visit labels in demos and corresponding numeric values in drugs
visitMapNum2Str = containers.Map(unique(drugs.visit), unique(demos.visit));

demos_drugs_matched = table();
for i = 1:length(drugs.subject)
    % Get the corresponding string value for the visit label in drugs
    visitStringValue = visitMapNum2Str(drugs.visit(i));
    
    % Find rows in demos that match the current subject ID and visit in drugs
    matchingRows_demos = demos(strcmp(demos.subject, drugs.subject{i}) & strcmp(demos.visit, visitStringValue), :);
    matchingRows_drugs = drugs(strcmp(drugs.subject, drugs.subject{i}) & drugs.visit == drugs.visit(i), :);
    matchingRows_FD = mrd(strcmp(mrd.subject, drugs.subject{i}) & strcmp(mrd.visit, visitStringValue), :); % get FD values
    
    if size(matchingRows_demos, 1) == 0
        % disp(['No match in demos for ', drugs.subject{i}, ' visit ', num2str(drugs.visit(i))]);
    else
        % Rename variables in drugs to avoid conflicts
        matchingRows_drugs.Properties.VariableNames = strcat(matchingRows_drugs.Properties.VariableNames, '_2');
        
        % Append the matching rows to the filtered table
        demos_drugs_matched = [demos_drugs_matched; matchingRows_demos, matchingRows_drugs, table(matchingRows_FD.mrd, 'VariableNames', {'mrd'})];
    end
end

%%
% remove rows where fmri was not clean/not available
demos_drugs_matched(demos_drugs_matched.mrd == -1, :) = [];

%% get time-series and outliers for each group

TS = cell(height(demos_drugs_matched), 1);
TS_interp = cell(size(TS));
TS_gsr = cell(size(TS));
TS_gsr_interp = cell(size(TS));
outliers = cell(size(TS));

for i = 1:height(demos_drugs_matched)
    outliers{i} = load(['rsfmri_txt/', demos_drugs_matched.subject{i}, '_', demos_drugs_matched.visit{i}, '_outliers.txt']);
    
    ts = load(['rsfmri_txt/', demos_drugs_matched.subject{i}, '_', demos_drugs_matched.visit{i}, '_gm-timeseries.txt']);
    TS{i} = ts'; 
    
    ts_gsr = GSR_parcellated_timeseries(ts, 'tzo116plus', 1:109);
    TS_gsr{i, 1} = ts_gsr';

    mask = zeros(size(ts, 2), 1);
    mask(outliers{i}) = 1;
    TS_gsr_interp{i, 1} = naninterp(TS_gsr{i}, 'outliermask', mask);
    TS_interp{i} = naninterp(TS{i}, 'outliermask', mask);
end

%%
save NCANDA_all_data.mat demos_drugs_matched outliers TS TS_interp TS_gsr TS_gsr_interp

%% Calculate and save FC without GSR

FC = cell(size(TS));
for i = 1:size(TS, 1)
    ts = TS{i};
    ts(outliers{i}, :) = [];
    FC{i} = compute_corr_manual(ts);
end
save NCANDA_FC.mat FC

clear FC

%% Calculate and save FC with GSR

FC = cell(size(TS_gsr));
for i = 1:size(TS_gsr, 1)
    ts = TS_gsr{i};
    ts(outliers{i}, :) = [];
    FC{i} = compute_corr_manual(ts);
end

save NCANDA_FCgsr.mat FC

%% Save demographics and matched data

save NCANDA_demos.mat demos_drugs_matched