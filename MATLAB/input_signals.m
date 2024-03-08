%% Input signals
% Global settings
T       = 90;       % one tracking run is 90 seconds
T_ef    = 81.92;    % only last 81.92 seconds used

w_m     = 2*pi / T_ef;    % measurement base frequency [rad/s]

sf = 100;   % sampling frequency [hz]
dt = 1/sf;  % timestep [s]

t = 0:dt:T; % time vector [s]


% Target signal:
n_t     = [5 6 13 14 27 28 41 42 53 54 73 74 103 104 139 140 193 194 229 230]; % integer multiple
w_t     = n_t.*w_m; % frequency of each sine [rad/s]
A_t     = [0.51 0.49 0.34 0.33 0.16 0.15 0.08 0.08 0.06 0.05 ... % amplitude of each sine [deg]
           0.03 0.03 0.02 0.02 0.02 0.02 0.01 0.01 0.01 0.01]; 
phi_t1  = [5.68 0.83 0.54 1.14 2.93 2.83 6.02 1.74 3.90 0.74 ... % phase (set1) of each sine [rad]
           5.65 3.70 3.63 1.42 3.64 5.94 4.74 3.72 4.32 2.15];
phi_t2  = [3.99 4.35 5.35 5.92 3.84 3.48 4.99 4.97 4.26 1.00 ... % phase (set2) of each sine [rad]
           4.69 5.01 5.44 5.78 0.86 2.05 2.03 2.88 3.01 2.93];
phi_t3  = [3.92 1.42 5.17 1.57 4.05 6.24 4.04 2.71 1.71 4.03 ... % phase (set3) of each sine [rad]
           0.67 5.85 5.13 4.14 2.66 5.65 4.28 1.92 1.03 3.21];
phi_t4  = [6.00 5.23 4.75 6.28 2.88 1.23 1.21 0.29 1.88 4.62 ... % phase (set4) of each sine [rad]
           0.89 0.97 0.74 5.81 5.21 1.08 0.70 2.81 4.44 0.53];
phi_t5  = [4.39 5.77 4.93 4.23 4.01 2.39 2.91 0.46 2.56 2.08 ... % phase (set5) of each sine [rad]
           4.56 3.33 1.43 1.44 5.97 4.55 5.76 5.28 2.67 2.50];
phis_t  = [phi_t1; phi_t2; phi_t3; phi_t4; phi_t5]; % list of all phase sets
       

phi_t = phis_t(rand_t, :); % randomly select one of the five sets

f_t = zeros(1, length(t)); % empty f_t
for i = 1:length(n_t)
    sine = A_t(i) * sin(w_t(i)*t + phi_t(i)); % each individual sine
    f_t = f_t + sine; % sum of sines
end


% Disturbance signal:
n_d     = [2 3 9 10 22 23 36 37 49 50 69 70 97 99 135 136 169 170 224 225]; % integer multiple
w_d     = n_d.*w_m; % frequency of each sine [rad/s]
A_d     = [0.10 0.15 0.30 0.30 0.19 0.18 0.12 0.12 0.12 0.12 ... % amplitude of each sine [deg]
           0.15 0.15 0.20 0.20 0.28 0.28 0.37 0.37 0.55 0.56];
phi_d1  = [2.74 3.92 3.97 1.65 3.60 -1.33 4.76 0.18 2.55 0.23 ... % phase (set1) of each sine [rad]
           0.93 3.49 2.84 5.12 0.54 5.60 5.89 0.88 1.77 2.46];
phi_d2  = [-0.02 1.79 -1.60 2.60 2.30 -1.74 4.09 1.51 1.90 2.89 ... % phase (set2) of each sine [rad]
           3.47 3.97 3.30 5.88 5.29 5.59 2.83 6.03 3.84 0.69];
phi_d3  = [-0.90 -1.55 1.87 -0.02 0.30 2.04 4.02 0.27 3.28 4.65 ... % phase (set3) of each sine [rad]
           0.63 0.97 5.77 5.65 1.65 1.04 1.22 0.98 1.48 1.49];
phi_d4  = [-0.71 2.02 1.41 1.90 -0.55 0.02 0.29 0.88 0.08 4.93 ... % phase (set4) of each sine [rad]
           0.14 0.75 3.29 2.52 1.28 5.22 2.93 0.70 5.28 0.95];
phi_d5  = [-0.58 3.67 0.80 4.00 3.45 -0.58 -0.82 -0.65 0.58 1.96 ... % phase (set5) of each sine [rad]
           4.37 4.63 2.91 3.80 -0.02 4.45 1.34 3.49 0.36 1.99]; 
phis_d  = [phi_d1; phi_d2; phi_d3; phi_d4; phi_d5]; % list of all phase sets
       
rand_d = rand_t; % randi([1 5], 1, 1); % random number between 1 and 5
phi_d = phis_d(rand_d, :); % randomly select one of the five sets

f_d = zeros(1, length(t)); % empty f_d
for i = 1:length(n_d)
    sine = A_d(i) * sin(w_d(i)*t + phi_d(i)); % each individual sine
    f_d = f_d + sine; % f_d = sum of sines
end

% Create white noise signal
rng('default') % reset random seed
rng(rand_t) % set random seed to fofu_real, so that remnant realisation will be comparable for different training settings
f_w = normrnd(0, 1, 1, length(t))/sqrt(dt); % normrnd(mean, standard deviation, size1, size2)


% Output signal data
f_t = [t; f_t];
f_d = [t; f_d];
f_w = [t; f_w];

save signals/ft.mat f_t
save signals/fd.mat f_d
save signals/fw.mat f_w

disp('Saved new f_t, f_d, and f_w')

% figure
% plot(t, f_t(2,:))
% xlabel('Time [s]')
% ylabel('f_t [deg]')
% 
% figure
% plot(t, f_d(2,:))
% xlabel('Time [s]')
% ylabel('f_d [deg]')