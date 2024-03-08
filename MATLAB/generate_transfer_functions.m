%% Randomness over parameters
if randomise_parameters
    r_low   = 80;
    r_high  = 120;
else
    r_low   = 100;
    r_high  = 100;
end

K_v_r       = randi([r_low r_high])/100;    w_nm_r  = randi([r_low r_high])/100;
t_v_r       = randi([r_low r_high])/100;    z_nm_r  = randi([r_low r_high])/100;
T_lead_r    = randi([r_low r_high])/100;    K_m_r   = randi([r_low r_high])/100;
T_lag_r     = randi([r_low r_high])/100;    t_m_r   = randi([r_low r_high])/100;
K_n_r       = randi([r_low r_high])/100;    T_l_r   = randi([r_low r_high])/100;

%% Generate transfer functions
s = tf('s');

% Stick gain
K_s = 0.2865;

%Controlled dynamics
H_theta_de = 10.62 * (s + 0.99) / (s * (s^2 + 2.58*s + 7.61));

%Neuromuscular model
w_nm     = w_nm_dic(key)*w_nm_r; % neuromuscular frequency
z_nm     = z_nm_dic(key)*z_nm_r; % neuromuscular damping ratio
H_nm     = w_nm^2 / (s^2 + 2*z_nm*w_nm*s + w_nm^2); % neuromuscular system dynamics

%Visual response
K_v      = K_v_dic(key)*K_v_r; % human operator visual gain
T_lead   = T_lead_dic(key)*T_lead_r; % visual lead time constant
T_lag    = T_lag_dic(key)*T_lag_r; % visual lag time constant
t_v      = t_v_dic(key)*t_v_r; % human operator visual delay
H_p_v    = K_v*(T_lead*s + 1)^2 / (T_lag*s +1) * exp(-s*t_v) * H_nm; % human operator visual response

%Motion response
K_m      = K_m_dic(key)*K_m_r; % human operator motion gain
t_m      = t_m_dic(key)*t_m_r; % human operator motion delay
H_scc    = (0.11*s +1)/(5.9*s + 1); % semicircular canal dynamics
H_p_m    = s^2*H_scc*K_m*exp(-s*t_m)*H_nm; % human operator motion response 

%Remnant low-pass filter
K_n     = K_n_dic(key)*K_n_r;
T_l     = T_l_dic(key)*T_l_r;
H_n = K_n * 1 / (1 + T_l*s);
