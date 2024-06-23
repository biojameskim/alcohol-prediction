%% collect all scans and save
clear all

demos = readtable('tabular/demographics_short.csv');
drugs = readtable('tabular/cahalan_plus_drugs.csv');

sids = unique(drugs.subject); %find subjects we have cahalan for
demos = demos(ismember(demos.subject,sids) & strcmp(demos.arm,'standard') & ~isnan(demos.mri_restingstate_age),:); %filter demos table to show only subjects we have chalan for

%% match rows across tables based on visit code & get mean FD

mrd = readtable('tabular/MRD.csv'); %mean FD values

% Define the mapping between visit labels in demos and corresponding numeric values in drugs
% visitMapStr2Num = containers.Map(unique(demos.visit), unique(drugs.visit));
visitMapNum2Str = containers.Map(unique(drugs.visit), unique(demos.visit));



demos_drugs_matched = table();
for i = 1:length(drugs.subject)
    % Get the corresponding string value for the visit label in drugs
    visitStringValue = visitMapNum2Str(drugs.visit(i));
    
    % Find rows in demos that match the current subject ID and visit in drugs
    matchingRows_demos = demos(strcmp(demos.subject, drugs.subject{i}) & strcmp(demos.visit, visitStringValue), :);
    matchingRows_drugs = drugs(strcmp(drugs.subject, drugs.subject{i}) & drugs.visit == drugs.visit(i), :);
    matchingRows_FD = mrd(strcmp(mrd.subject,drugs.subject{i}) & strcmp(mrd.visit,visitStringValue), :); %get FD values
    
    % note if we have preprocessed dmri
    dmri_yn = exist(['dti_mat/',drugs.subject{i},'_',visitStringValue,'.mat'],'file');
    if dmri_yn == 2
        dmri_yn = 1;
    end
    
    if size(matchingRows_demos,1) == 0
%         disp(['No match in demos for ', drugs.subject{i}, ' visit ', num2str(drugs.visit(i))]);
    else
        % Rename variables in drugs to avoid conflicts
        matchingRows_drugs.Properties.VariableNames = strcat(matchingRows_drugs.Properties.VariableNames, '_2');
        
        % Append the matching rows to the filtered table
        demos_drugs_matched = [demos_drugs_matched; matchingRows_demos, matchingRows_drugs, table(matchingRows_FD.mrd,'VariableNames',{'mrd'}), table(dmri_yn,'VariableNames',{'dmri'})];
    end
end

%%
% remove rows where fmri was not clean/not available
demos_drugs_matched(demos_drugs_matched.mrd==-1,:)=[];

% remove rows where dmri was not clean/available
demos_drugs_matched(demos_drugs_matched.dmri==0,:)=[];

%% get time-series and outliers and dMRI for each group

sri124 = readtable('tabular/sri24_parc116_gm.csv');

SC = cell(height(demos_drugs_matched),1);
outliers = cell(size(SC));
TS = cell(size(SC));
TS_interp = cell(size(SC));
TS_gsr = cell(size(SC));
TS_gsr_interp = cell(size(SC));

for i=1:height(demos_drugs_matched)
    
    SC{i} = load(['dti_mat/',demos_drugs_matched.subject{i},'_',demos_drugs_matched.visit{i},'.mat']);
    outliers{i} = load(['rsfmri_txt/',demos_drugs_matched.subject{i},'_',demos_drugs_matched.visit{i},'_outliers.txt']);
    
    ts = load(['rsfmri_txt/',demos_drugs_matched.subject{i},'_',demos_drugs_matched.visit{i},'_gm-timeseries.txt']);
    TS{i} = ts'; 
    
    ts_gsr = GSR_parcellated_timeseries(ts,'tzo116plus',1:109);
    TS_gsr{i,1} = ts_gsr';
    
    mask = zeros(size(ts,2),1);
    mask(outliers{i}) = 1;
    TS_gsr_interp{i,1} = naninterp(TS_gsr{i},'outliermask',mask);
    TS_interp{i} = naninterp(TS{i},'outliermask',mask);
    
end

%%
save NCANDA_all_data.mat demos_drugs_matched SC outliers TS TS_interp TS_gsr TS_gsr_interp sri124

%%

save NCANDA_SC.mat SC
%%

for i=1:size(TS,1)
    ts = TS{i};
    ts(outliers{i},:) = [];
    FC{i} = compute_corr_manual(ts);
end
save NCANDA_FC.mat FC

%%
clear FC

for i=1:size(TS_gsr,1)
    ts = TS_gsr{i};
    ts(outliers{i},:) = [];
    FC{i} = compute_corr_manual(ts);
end

save NCANDA_FCgsr.mat FC

%% 

save NCANDA_demos.mat demos_drugs_matched