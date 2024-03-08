%% Specify constants
% -----------settings--------------
% Set keys -ENTER KEYS HERE:-
group = 'NM';    % ['NM' / 'M'] 
phase = 'TRAIN'; % ['TRAIN' / 'EVAL']
run   = 'END'; % ['START' / 'END']

generate_data = false; % [true / false] if set to true will save .csv files 
N_runs = 5;    % number of .csv files to be saved with current setting
randomise_parameters = true; % [true / false] if set to true will apply randomness to pilot model parameters
% ---------------------------------

key = append(group,phase,run); % key which will retrieve values for model parameters

disp(['Running ' group ' ' phase ' ' run])

% Enable or disable motion reponse loop based on settings
if (strcmp(group,'M') && strcmp(phase,'TRAIN')) || (strcmp(group,'NM') && strcmp(phase,'EVAL'))
    Mot_Resp = true; % Motion response is enabled in simulink pilot model
else
    Mot_Resp = false; % Motion response is disabled in simulink pilot model
end

%% Creating dictionaries
% Dictionaries with all results found from *
groups  = [{'NM'} {'M'}];
phases  = [{'TRAIN'} {'EVAL'}];
runs    = [{'START'} {'END'}];

%Create empty keys
keySet = cell(1,4);

%Fill all keys
c = 1;
for i=1:2
    for j=1:2
        for k=1:2
            keySet(c) = append(groups(i),phases(j),runs(k));  
            c = c+1;
        end
    end
end
% Human pilot model parameters for different keys:
%                               KEYS:
%                      NM         |         M
%                train  |   eval  |  train  |   eval
%              start|end|start|end|start|end|start|end
K_v_set     = [2.40 3.05 3.50 4.25 3.00 4.75 3.80 4.25]; % Visual gains, taken from *
t_v_set     = [0.28 0.21 0.25 0.23 0.27 0.22 0.21 0.18]; % Visual delays, taken from *
T_lead_set  = [0.66 0.44 0.40 0.30 0.50 0.30 0.44 0.41]; % Visual lead time constants, taken from *
T_lag_set   = [3.00 1.40 1.23 0.70 1.85 0.95 1.60 1.50]; % Visual lag time constants, taken from *
w_nm_set    = [10   10   11   11   15   14   11   11  ]; % Neuromuscular frequencies, taken from *
z_nm_set    = [0.50 0.25 0.25 0.25 0.50 0.30 0.21 0.2 ]; % Neuromuscular damping ratios, taken from *
K_m_set     = [0    0    1.10 3.05 0.90 3.00 0    0   ]; % Motion gains, taken from *
t_m_set     = [0    0    0.23 0.18 0.30 0.19 0    0   ]; % Motion delays, taken from *
K_n_set     = [0.42 0.28 0.0  0.0  0.0  0.0  0.0  0.0 ]; % H_n gain, remnant lowpass filter 1.04 0.44
T_l_set     = [0.59 0.46 0.0  0.0  0.0  0.0  0.0  0.0 ]; % H_n time delay, remnant lowpass filter 0.75 0.36
% * Effects of Simulator Motion Feedback on Training of Skill-Based Control Behavior (2016)

% Create dictionaries
K_v_dic     = containers.Map(keySet,K_v_set);     w_nm_dic  = containers.Map(keySet,w_nm_set);
t_v_dic     = containers.Map(keySet,t_v_set);     z_nm_dic  = containers.Map(keySet,z_nm_set);
T_lead_dic  = containers.Map(keySet,T_lead_set);  K_m_dic   = containers.Map(keySet,K_m_set);
T_lag_dic   = containers.Map(keySet,T_lag_set);   t_m_dic   = containers.Map(keySet,t_m_set);
K_n_dic     = containers.Map(keySet,K_n_set);     T_l_dic   = containers.Map(keySet,T_l_set);

%% Generate transfer functions
generate_transfer_functions; % calculate all transfer functions (run generate_transfer_functions.m)

%% Create signals and run simulink simulation
% set random number for realisation of target/disturbance signal
rand_t = randi([1 5], 1, 1); % random number between 1 and 5
% rand_t = 5
% disp('input_signals.m: fofu real fixed to 5!!!')

if generate_data
    for run=1:N_runs
        input_signals; % generate f_t and f_d (run input_signals.m)
        generate_transfer_functions; % called again in this loop so that transfer functions are slightly different every time (if randomise_parameters is set to true)
        
        % r_seed = randi([1 1000]); % Random seed for remnant
        out = sim('pilot_data_generator',T); % run simulink pilot model, T inherited from input_signals.m
        disp('Simulated pilot compensatory tracking task') 
        
        % write simulated data to output vector
        output = zeros(length(out.e), 3);
        output(:,1) = t;
        output(:,2) = -out.u; % change sign to replicate 'PCTRLS uy'
        output(:,3) = out.e;
        output = array2table(output);
        output.Properties.VariableNames(1:3) = {'t','u','e'};

        path = 'pilot_data/generated/';
        filename = append(key, '-run-', string(run), '.csv');
        disp('Writing to:')
        disp(append(path,filename))
        writetable(output,append(path,filename));
    end
    
else 
    data = cell(N_runs, 1);
    for run=1:N_runs
        input_signals; % generate f_t and f_d (run input_signals.m)
       
        generate_transfer_functions; % called again in this loop so that transfer functions are slightly different every time (if randomise_parameters is set to true)
        
        r_seed = randi([1 1000]); % Random seed for remnant
        out = sim('pilot_data_generator',T); % run simulink pilot model, T inherited from input_signals.m
        disp('Simulated pilot compensatory tracking task') 
        
        % shaping data same way as the actual pilot data
        output = zeros(length(out.e), 6);
        output(:,1) = t;
        output(:,2) = f_t(2, :)/(180/pi); % store as radians
        output(:,3) = f_d(2, :)/(180/pi);
        output(:,5) = out.e/(180/pi);
        output(:,6) = out.u/3.490402474467116/(180/pi); % DYN u 
        output(:,11) = -out.u/(180/pi); % change sign to replicate 'PCTRLS uy'
        
        % filter out first 8.08 sec and last 5 sec
        filter_l = t>=8.08; filter_u = t<=90; % lower limit and upper limit for t
        filter = logical(filter_l.*filter_u); % calculate boolean filter
            
        data{run} = output(filter, :); % apply filter and store data in cell
    end
end



% %% temp 
% 
% input_signals; % generate f_t and f_d
% 
% r_seed = randi([1 1000]); % Random seed for remnant
% out = sim('pilot_data_generator',T); % run simulink pilot model, T inherited from input_signals.m
% disp('Simulated pilot compensatory tracking task') 
% disp('Error variance: ')
% disp(var(out.e))
% disp('Input variance: ')
% disp(var(out.u)) 
