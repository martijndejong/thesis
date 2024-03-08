%% Loading data
% load in human parameters from:
% Effects of Simulator Motion Feedback on Training of Skill-Based Control Behavior (2016)
load Parameters/Results.mat
load Parameters/new_remnant5fdfd_parameters.mat
parameters = Data;

% Replacing parameter values that cause model instability:
parameters.NM.Enm.Meas.Subject_2.training(1)=1.0; % is originally 23.9569
parameters.NM.Enm.Meas.Subject_4.training(16)=1.0; % is originally 4.4166
parameters.NM.Enm.Meas.Subject_6.training(5)=1.0; % is originally 285.3621
parameters.NM.Wnm.Meas.Subject_6.training(5)=10.0; % is originally 8875.1867
parameters.NM.Enm.Meas.Subject_7.training(6)=1.0; % is originally 5171.5815
parameters.NM.Wnm.Meas.Subject_7.training(6)=10.0; % is originally 17525.7016
parameters.NM.Enm.Meas.Subject_9.training(27)=1.0; % is originally 8.7690
parameters.NM.Enm.Meas.Subject_9.training(43)=1.0; % is originally 20.9018
parameters.NM.Wnm.Meas.Subject_11.training(30)=10.0; % is originally 59.5303

% remnant_parameters.NM.T.Subject_5.training(48)= 0.2; % is originally 2.1538e-09
% remnant_parameters.NM.T.Subject_8.training(77)= 0.2; % is originally 4.9241e-06
% remnant_parameters.NM.T.Subject_8.training(82)= 0.2; % is originally 1.5972e-06
% remnant_parameters.NM.T.Subject_9.training(68)= 0.2; % is originally 1.5972e-06
% remnant_parameters.NM.T.Subject_10.training(89)= 0.2; % is originally 1.5972e-06
% remnant_parameters.NM.T.Subject_11.training(46)= 0.2; % is originally 1.5972e-06
% remnant_parameters.NM.T.Subject_11.training(62)= 0.2; % is originally 1.5972e-06
% remnant_parameters.NM.T.Subject_11.training(81)= 0.2; % is originally 1.5972e-06
% remnant_parameters.NM.T.Subject_12.training(52)= 0.2; % is originally 1.5972e-06
% remnant_parameters.NM.T.Subject_12.training(76)= 0.2; % is originally 1.5972e-06
% remnant_parameters.NM.T.Subject_12.training(79)= 0.2; % is originally 1.5972e-06


%% User input
% set group and phase to be simulated
group       = 'NM';         %'NM'/'M'
phase       = 'training';   %'training'/'test'
save_output = true;        % false/true
K_r         = 0.0;          %0.0 -- 1.0 remnant gain


%% Simulation settings based on user input
% Enable or disable motion reponse loop based on settings
if (strcmp(group,'M') && strcmp(phase,'training')) || (strcmp(group,'NM') && strcmp(phase,'test'))
    Mot_Resp = true; % Motion response is enabled in simulink pilot model
else
    Mot_Resp = false; % Motion response is disabled in simulink pilot model
end


%% load in parameters for subject
for subject = 1:12

    sub_str = append('Subject_', string(subject));

    % neuromuscular parameters
    w_nm_list   = parameters.(group).Wnm.Meas.(sub_str).(phase);
    z_nm_list   = parameters.(group).Enm.Meas.(sub_str).(phase);
    % visual response parameters
    K_v_list    = parameters.(group).Kv.Meas.(sub_str).(phase);
    T_lead_list = parameters.(group).TLead.Meas.(sub_str).(phase);
    T_lag_list  = parameters.(group).TLag.Meas.(sub_str).(phase);
    t_v_list    = parameters.(group).te.Meas.(sub_str).(phase);
    % motion response parameters
    K_m_list    = parameters.(group).Km.Meas.(sub_str).(phase);
    t_m_list    = parameters.(group).tm.Meas.(sub_str).(phase);
    % remnant parameters
%     K_n_list    = rem_par.K.(sub_str);
%     T_l_list    = rem_par.T.(sub_str);
    K_n_list = remnant_parameters.(group).K.(sub_str).(phase);
    T_l_list = remnant_parameters.(group).T.(sub_str).(phase);


    % run info parameters from 'SubjectRuns' to read run# and fofureal
    if strcmp(group,'NM') grp_n = 1;, else grp_n = 2;, end
    info_file   = char('Score_Group'+string(grp_n)+'_Subject'+string(subject)+'.dat'); % generate filename

    run_info    = HDRLOAD(info_file); % read file 

    filter      = run_info(:,2) == Mot_Resp; % filter to only look at training/testing phase
    run_info    = run_info(filter,:);

    run_list    = run_info(:,1); % list of runs that are were marked as successful
    fofu_list   = run_info(:,3); % list of corresponding fofureal indices

    %% Run pilot simulation 
    for run_id = 1:length(run_list)
        disp('for reference sub: '+string(subject))
        disp('for reference run: '+string(run_id))

        % Generate same target/disturbance signal as actual pilot
        rand_t = fofu_list(run_id); % find correct fofureal index
        input_signals; % generate corresponding target/disturbance signal
        
        % Generate transfer functions of actual pilot based on run_id
        generate_transfer_functions_actual_parameters;

        % Generate tracking data using the actual pilot settings
        out = sim('pilot_data_generator',T); % run simulink pilot model, T inherited from input_signals.m
        disp('Simulated pilot compensatory tracking task')   

        % write simulated data to output vector
        output = zeros(length(out.e), 6);
        output(:,1) = t;
        output(:,2) = f_t(2,:)/(180/pi); %convert to radians;
        output(:,3) = f_d(2,:)/(180/pi); %convert to radians;
        output(:,4) = -out.e/(180/pi); % change signs to replicate 'e', convert to radians
        output(:,5) = out.u/3.490402474467116/(180/pi); % divide by factor to make 'DYN u'
        output(:,6) = -out.u/(180/pi); % change sign to replicate 'PCTRLS uy'   
        output(:,7) = out.x/(180/pi); %convert to radians;

%         % filter out first 8.08 sec and last 5 sec
%         filter_l = t>=8.08; filter_u = t<=90; % lower limit and upper limit for t
%         filter = logical(filter_l.*filter_u); % calculate boolean filter
%         output = output(filter, :); % apply filter

        % store output in table
        output = array2table(output);
        output.Properties.VariableNames(1:7) = {'t','ft','fd','e','DYN u','PCTRLS uy', 'DYN x'};
        
        
        % Generate file name that is consistent with actual pilot data
        run_name = sprintf( '%03d', run_list(run_id));
        if Mot_Resp mot_name = 'ON';, else mot_name = 'OFF';, end
        filename = 'ToTPitchTrackingData_Group'+string(grp_n)...
                   +'_Subject'+string(subject)...
                   +'_nr'+run_name...
                   +'_mot'+mot_name...
                   +'_fofuReal'+string(rand_t)...
                   +'.csv';
               
        if save_output
            path = 'data/generated5fdfd-times0.0/';
            disp('Writing to:')
            disp(append(path,filename))
            writetable(output,append(path,filename));
        end
    end
end


