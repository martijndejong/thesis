%% Generate transfer functions
s = tf('s');

% Stick gain
K_s = 0.2865;

%Controlled dynamics
H_theta_de = 10.62 * (s + 0.99) / (s * (s^2 + 2.58*s + 7.61));

%Neuromuscular model
w_nm     = w_nm_list(run_id); % neuromuscular frequency
z_nm     = z_nm_list(run_id); % neuromuscular damping ratio
H_nm     = w_nm^2 / (s^2 + 2*z_nm*w_nm*s + w_nm^2); % neuromuscular system dynamics

%Visual response
K_v      = K_v_list(run_id); % human operator visual gain
T_lead   = T_lead_list(run_id); % visual lead time constant
T_lag    = T_lag_list(run_id); % visual lag time constant
t_v      = t_v_list(run_id); % human operator visual delay
H_p_v    = K_v*(T_lead*s + 1)^2 / (T_lag*s +1) * exp(-s*t_v) * H_nm; % human operator visual response

%Motion response
K_m      = K_m_list(run_id); % human operator motion gain
t_m      = t_m_list(run_id); % human operator motion delay
H_scc    = (0.11*s +1)/(5.9*s + 1); % semicircular canal dynamics
H_p_m    = s^2*H_scc*K_m*exp(-s*t_m)*H_nm; % human operator motion response 

%Remnant low-pass filter
% if run_id<=50 K_n = K_n_list(1);, else K_n = K_n_list(2);, end;  %0.42 0.28  -> Results.mat uitbreiden met deze getallen (per 5 runs)
% if run_id<=50 T_l = T_l_list(1);, else T_l = T_l_list(2);, end;  %0.59 0.46
K_n = K_n_list(run_id);
T_l = T_l_list(run_id);
H_n = K_n * 1 / (1 + T_l*s);
