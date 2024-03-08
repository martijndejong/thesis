% clear all
%% Calculate S_uu for multiple runs

% enter filter specifications
group_list = [1]; % group 1 or group 2
subject_list = [1]; % what subjects to inspect
run_list = [6:10]; %[96:100]; % what runs to inspect
motion_list = [0]; % motion off 0, motion on 1
fofu_list = [1:5];%[1:5]; % realisation of ft and fd

% load multiple runs into data cell
data = HDRLOAD_list(group_list, subject_list, run_list, motion_list, fofu_list); % group_list, subject_list, run_list, motion_list, fofu_list
% data = HDRLOAD_list_sim(group_list, subject_list, run_list, motion_list, fofu_list);

% Sampling frequency and dt
fs = 100;  % hz
dt = 1/fs; % s

F_u_list  = zeros(4096, length(data)); % list to store different F_u's
S_uu_list = zeros(4096, length(data)); % list to store different S_uu's
% calculate S_uu for each loaded run, plot, and append

for i = 1 : length(data)
    t = data{i}(:,1);           % read time vector from dataset
    u = data{i}(:, 11)*(180/pi); % read input from dataset  % 6:DYN u , 11:PCTRLS uy 
    
    % TEMP code for figuring out what input to use
    u_DYN = data{i}(:, 6);
    u_PC  = data{i}(:, 11);
    
    u_DYN_v(i) = var(u_DYN);
    u_PC_v(i) = var(u_PC);
    % TEMP code for figuring out what input to use
    
    ft = data{i}(:,2)*(180/pi);          % read target signal from dataset
    fd = data{i}(:,3)*(180/pi);          % read disturbance signal from dataset
    
    disp(i)
    if length(t)<8192
        disp('t is too short!!')
    
    else
        T = max(t);     % tracking length
        N = length(t);  % number of datapoints

        F_u = fft(u); % fourier transform

        S_uu = dt*(1/N)*(F_u.*conj(F_u)); % power spectral density function
        S_uu = S_uu(1:round(N/2)-1); % single sided spectrum

        omega = 2*pi*fs*(0:(N/2)-1)/N; % frequencies in rad/s to plot PSD

        % add individual S_uu to list
        S_uu_list(:,i) = S_uu;
        
        % add individual F_u to list
        F_u_mag         = 2*abs(F_u(1:round(N/2)-1))/N ;
        %F_u_mag(1:end)  = 2*F_u_mag(1:end);
        F_u_list(:,i)   = F_u_mag;

%         % plot individual F_u
%         plot(omega, F_u, 'color', [0.8 0.2 0.2])
%         hold on    
        
%         % calculate S_fdfd
%         FT = fft(fd);
%         S_ftft = dt*(1/N)*(FT.*conj(FT)); % power spectral density function
%         S_ftft = S_ftft(1:round(N/2)-1); % single sided spectrum
    end
end

% start figure with plot config
figure
hold on
legend on
legend('location', 'southwest')
LineWidth = 1.3;

%plot(omega, abs(F_u(1:round(N/2)-1).^2)/T, 'LineWidth', 3)

% plot individual S_uu's
name = append('S_{uu} individual runs (', string(i), 'x)');
p = plot(omega, S_uu_list, 'color', [0.8 0.8 0.8],'LineWidth',LineWidth); % 'DisplayName', name,
legend([p(1)], name);

% take average S_uu and plot
S_uu_mean = mean(S_uu_list, 2);
plot(omega, S_uu_mean, 'color', [0.1 0.1 0.1], 'DisplayName', 'S_{uu} five-run average','LineWidth',LineWidth);

%% Plot frequency and magnitude of target and disturbance signal
% measurement time and measurement base frequency
T_ef    = 81.92;          % only last 81.92 seconds used
w_m     = 2*pi / T_ef;    % measurement base frequency [rad/s]

% Target signal frequencies and magnitudes:
n_t     = [5 6 13 14 27 28 41 42 53 54 73 74 103 104 139 140 193 194 229 230]; % integer multiple
w_t     = n_t.*w_m; % frequency of each sine [rad/s]
A_t     = [0.51 0.49 0.34 0.33 0.16 0.15 0.08 0.08 0.06 0.05 ... % amplitude of each sine [deg]
           0.03 0.03 0.02 0.02 0.02 0.02 0.01 0.01 0.01 0.01]; 
       
% S_uu amplitude at target signal, to plot circle around it
A_t_circ = zeros(length(A_t),1);
for i = 1:length(w_t)
    w_t_i = w_t(i);
    [d, ix] = min(abs(omega-w_t_i));
    A_t_circ(i) = S_uu_mean(ix);
end
       
% Disturbance signal frequencies and magnitudes:
n_d     = [2 3 9 10 22 23 36 37 49 50 69 70 97 99 135 136 169 170 224 225]; % integer multiple
w_d     = n_d.*w_m; % frequency of each sine [rad/s]
A_d     = [0.10 0.15 0.30 0.30 0.19 0.18 0.12 0.12 0.12 0.12 ... % amplitude of each sine [deg]
           0.15 0.15 0.20 0.20 0.28 0.28 0.37 0.37 0.55 0.56];
       
% S_uu amplitude at target signal, to plot circle around it
A_d_circ = zeros(length(A_d),1);
for i = 1:length(w_d)
    w_d_i = w_d(i);
    [d, ix] = min(abs(omega-w_d_i));
    A_d_circ(i) = S_uu_mean(ix);
end

% --- Plotting ---
%Plot TARGET magnitudes and frequencies
scatter(w_t, A_t_circ, 'DisplayName', 'S_{uu_{t}}, target frequency', 'MarkerEdgeColor', '#D95319','LineWidth',LineWidth) 
%Plot DISTURBANCE magnitude and frequencies
scatter(w_d, A_d_circ, 'DisplayName', 'S_{uu_{d}}, disturbance frequency', 'MarkerEdgeColor', '#EDB120','LineWidth',LineWidth)


%% Compute variance of the countrol output Fourier coefficients [=S_uu_n]
%------old method for S_uu_n--------------
%S_uu_n = var(F_u_list, 0, 2)/dt;
%scatter(omega, S_uu_n,'DisplayName', 'M2: S_{uu_{n}} = \sigma^{2}_{U}', 'marker','x','MarkerEdgeColor', '#0072BD','LineWidth',LineWidth)
%-----------------------------------------

%------new method: interpolation----------
% NEW (SIMPLE) METHOD S_uu_n!!!: take S_uu_n as all behavior outside of target / disturbance frequencies
wt_wd       = sort([w_t w_d]); % list all frequencies with target/disturbance signal
S_uu_n_s    = S_uu_mean; % set S_uu_n = S_uu and then remove entries where there is f_t/f_d
omega_n     = omega;     % set omega_remnant = omega and then remove entries where there is f_t/f_d
ix_list = zeros(length(wt_wd),1);
for i = 1:length(wt_wd)
    % find omega index closest to w_t's and w_d's
    [d, ix] = min(abs(omega-wt_wd(i)));
    % store indices that must be removed
    ix_list(i) = ix;
end
% remove those elements from list
omega_n(ix_list)     = [];
S_uu_n_s(ix_list)    = [];

% interpolate to get values for S_uu_n at target/disturbance frequency
S_uu_n_s_pol = interp1(omega_n, S_uu_n_s, omega);
scatter(omega, S_uu_n_s_pol, 'marker','x','MarkerEdgeColor', '#0072BD', 'DisplayName', 'S_{uu_{n}}, interpolated ','LineWidth',LineWidth)

% overwrite old method with new method
S_uu_n = S_uu_n_s_pol;


% more plotting settings
set(gca,'xscale','log') % set xscale to log
set(gca,'yscale','log') % set yscale to log

xlabel('\omega, rad/s')
ylabel('S_{uu} deg^{2} / Hz')

xlim([0.04 300])
ylim([10^-9.5 10^2])
%hold off

%% Calculate power of remnant
% var(u_n) / var(u) = integral(S_uu_n)/integral(S_uu)
% calculate relative contribution of remnant; var(u_n) / var(u) = integral(S_uu_n)/integral(S_uu)
rem_cont  = trapz(omega, S_uu_n) / trapz(omega, S_uu_mean);


%hold off
%% Calculate S_nn_e(j wt) 
% Retreive S_uu_n and S_uu_t for each jwt
S_uu_n_wt    = zeros(length(w_t), 1);
S_uu_t      = zeros(length(w_t), 1);
% S_ftft_wt = zeros(length(w_t), 1);
for i = 1:length(w_t)
    w_t_i = w_t(i);
    [d, ix] = min(abs(omega-w_t_i));
    S_uu_n_wt(i)     = S_uu_n(ix);
    S_uu_t(i)       = S_uu_mean(ix);
%     S_ftft_wt(i)    = S_ftft(ix);
end
%QUESTION: S_ftft = A_t? OR PERFORM SPECTRAL ANALYSIS ON fd SIGNAL?
%ANSWER: can use A_t as follows:
F_t = A_t/2*N;
S_ftft_wt = (F_t.^2*dt/N)';

S_nn_e = S_uu_n_wt./S_uu_t.*S_ftft_wt;

%---figures-progress-meeting-7------
%scatter(w_t, S_uu_n_wt, 'DisplayName', 'S_{uu_{n}}(j \omega_{t})', 'MarkerEdgeColor', '#00FF00','LineWidth', 1.5)
%scatter(w_t, S_uu_t, 'DisplayName', 'S_{uu_{t}}(j \omega_{t})',  'MarkerEdgeColor', '#FF00FF' ,'LineWidth', 1.5)
%scatter(w_t, S_ftft_wt, 'DisplayName', 'S_{ftft}(j \omega_{t})',  'MarkerEdgeColor', '#00FFFF' ,'LineWidth', 1.5)
%---------------------------------

% % Noise low-pass filter
% s = tf('s');
% K = 0.095;
% T_l = 0.23;
% H_n = K/(1+T_l*s);

% Fitting model
xdata = w_t;
ydata = S_nn_e';

fun = @(x,xdata)x(1)^2*(1+x(2)^2*xdata.^2).^-1;

x0 = [0.5, 0.5];
x = lsqcurvefit(fun,x0,xdata,ydata)


figure
hold on
legend on
plot(omega, fun(x, omega), 'DisplayName', 'model fit', 'LineWidth',LineWidth)
scatter(w_t, S_nn_e, 'DisplayName', 'S_{nn_{e}}(j\omega_{t})', 'LineWidth',LineWidth)



% more plotting settings
xlim([0.7*10^-1 10^2])
ylim([10^-5 10^0])

set(gca,'xscale','log') % set xscale to log
set(gca,'yscale','log') % set yscale to log

xlabel('\omega, rad/s')
ylabel('S_{nn_{e}} deg^{2} / Hz')


hold off
