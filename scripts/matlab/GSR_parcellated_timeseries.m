function[ts_clean] = GSR_parcellated_timeseries(roi_ts, atlas_file, labellist)
% GSR from parcellated timeseries
%Inputs: 
%roi_ts is NumROI x NumTimepoints matrix (note that this becomes transposed later
 %atlas_file is the name of your parcellation NIFTI (NOTE: location of this
 %file is hardcoded on line 18)
%labellist is a vector if which ROIs you care about (as numbers); omit this
%to have the full parcellation (default)

%Outputs: ts_clean the GSR-d timeseries, in NxT format (i.e same format as
%the original input)

%% SCRIPT

roi_ts=roi_ts.'; %transpose to 220x90 - i.e. TxN

%Load the atlas NIFTI - will need to customise this to your directories
atlas_fullpath = ['/Users/sps253/Documents/parcs/', atlas_file];
nii = load_untouch_nii([atlas_fullpath, '.nii']);
voxel_atlas = nii.img;

if ~exist('labellist', 'var')
    labellist = setxor(unique(voxel_atlas), 0); %exclude zero
end

%make a Nx1 list
labelsize=zeros(numel(labellist),1); 

for i = 1:numel(labellist)
    labelsize(i)=sum(voxel_atlas(:)==labellist(i)); %how many voxels have that label value
end

mean_ts=roi_ts * labelsize/sum(labelsize); %an ROI-size weighted average of the time series
confounds=[mean_ts [0; diff(mean_ts)] ones(size(mean_ts))];
Q=eye(size(confounds,1))-confounds*pinv(confounds); %al857: regress out the confounds
ts_clean=transpose(Q*roi_ts);
end


%% ORIGINAL CODE
% roi_ts=load('somefile.mat'); %90x220 ROI time series
% roi_ts=roi_ts.'; %transpose to 220x90
% Vaal=read_avw('AAL.nii.gz');
% labellist=[1,7,12,....]; %or whatever the 1x90 list of label values that you have
% labelsize=zeros(numel(labellist),1); %make a 90x1 list
% for i = 1:numel(labellist)
% labelsize(i)=sum(Vaal(:)==labellist(i)); %how many voxels have that label value
% end
% mean_ts=roi_ts * labelsize/sum(labelsize); %an ROI-size weighted average of the time series
% confounds=[mean_ts [0; diff(mean_ts)] ones(size(mean_ts))];
% Q=eye(size(confounds,1))-confounds*pinv(confounds);ts_clean=Q*roi_ts;
