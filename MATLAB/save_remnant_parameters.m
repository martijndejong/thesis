%% Loading data
% load in human parameters from:
% Effects of Simulator Motion Feedback on Training of Skill-Based Control Behavior (2016)

%% User input
% set group and phase to be simulated
group       = 'NM';         %'NM'/'M'
phase       = 'training';   %'training'/'test'
save_output = true;        % false/true


%% Simulation settings based on user input
% Enable or disable motion reponse loop based on settings
if (strcmp(group,'M') && strcmp(phase,'training')) || (strcmp(group,'NM') && strcmp(phase,'test'))
    Mot_Resp = true; % Motion response is enabled in simulink pilot model
else
    Mot_Resp = false; % Motion response is disabled in simulink pilot model
end


%% Loop subjects
for subject = 1:13

    sub_str = append('Subject_', string(subject));

    % run info parameters from 'SubjectRuns' to read run# and fofureal
    if strcmp(group,'NM') grp_n = 1;, else grp_n = 2;, end
    info_file   = char('Score_Group'+string(grp_n)+'_Subject'+string(subject)+'.dat'); % generate filename

    run_info    = HDRLOAD(info_file); % read file 

    filter      = run_info(:,2) == Mot_Resp; % filter to only look at training/testing phase
    run_info    = run_info(filter,:);

    run_list    = run_info(:,1); % list of runs that are were marked as successful
    fofu_list   = run_info(:,3); % list of corresponding fofureal indices

    %% Run pilot simulation 
    for run_id = 1:5:length(run_list)
        disp('for reference sub: '+string(subject))
        disp('for reference run: '+string(run_id))

        calculate_remnant_parameters
        
        remnant_parameters.(group).K.(sub_str).(phase)(run_id:run_id+4,1) = x(1);
        remnant_parameters.(group).T.(sub_str).(phase)(run_id:run_id+4,1) = x(2);
        
    end
end

save Parameters/sept_remnant5fdfd_parameters.mat remnant_parameters
