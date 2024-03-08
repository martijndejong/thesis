%% Calculate S_uu for multiple runs
% load multiple runs into data cell

% TEMP
run_id=1;
grp_n = 1;
subject = 1;
run_list = 96:100;
Mot_Resp = 0;
fofu_list = 1:5;

data = HDRLOAD_list(grp_n, subject, run_list(run_id:run_id+4 ), Mot_Resp, fofu_list(run_id:run_id+4 )); % group_list, subject_list, run_list, motion_list, fofu_list
                                           %:run_id+4                   %:run_id+4


% Sampling frequency and dt
fs = 100;  % hz
dt = 1/fs; % s

F_u_list  = zeros(4096, length(data)); % list to store different F_u's
S_uu_list = zeros(4096, length(data)); % list to store different S_uu's
% calculate S_uu for each loaded run, plot, and append

for i = 1 : length(data)
    t_rem = data{i}(:,1);           % read time vector from dataset
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
    if length(t_rem)<8192
        disp('t is too short!!')
        % generate artificial time vector
        A = 8.08; % starting number
        S = 0.01; % step
        N = 8193; % number of values
        V = A+S*(0:N-1);
        t_rem = V;
        % repeat signals to be same length as time vector
        u = [u; u; u]; % too lengthy u
        u = u(1:length(t_rem)); % sliced u
        ft = [ft; ft; ft]; % too lengthy ft
        ft = ft(1:length(t_rem)); % sliced ft
        fd = [fd; fd; fd]; % too lengthy fd
        fd = fd(1:length(t_rem)); % sliced fd
    end
        
    T = max(t_rem);     % tracking length
    N = length(t_rem);  % number of datapoints

    F_u = fft(u); % fourier transform

    S_uu = dt*(1/N)*(F_u.*conj(F_u)); % power spectral density function
    S_uu = S_uu(1:round(N/2)-1); % single sided spectrum
    disp('reached')
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

S_uu_mean = mean(S_uu_list, 2);

%% Plot frequency and magnitude of target and disturbance signal
% measurement time and measurement base frequency
T_ef    = 81.92;          % only last 81.92 seconds used
w_m     = 2*pi / T_ef;    % measurement base frequency [rad/s]

% Target signal frequencies and magnitudes:
n_t     = [5 6 13 14 27 28 41 42 53 54 73 74 103 104 139 140 193 194 229 230]; % integer multiple
w_t     = n_t.*w_m; % frequency of each sine [rad/s]
A_t     = [0.51 0.49 0.34 0.33 0.16 0.15 0.08 0.08 0.06 0.05 ... % amplitude of each sine [deg]
           0.03 0.03 0.02 0.02 0.02 0.02 0.01 0.01 0.01 0.01]; 
       
      
% Disturbance signal frequencies and magnitudes:
n_d     = [2 3 9 10 22 23 36 37 49 50 69 70 97 99 135 136 169 170 224 225]; % integer multiple
w_d     = n_d.*w_m; % frequency of each sine [rad/s]
A_d     = [0.10 0.15 0.30 0.30 0.19 0.18 0.12 0.12 0.12 0.12 ... % amplitude of each sine [deg]
           0.15 0.15 0.20 0.20 0.28 0.28 0.37 0.37 0.55 0.56];
% correct amplitudes
A_d_cor = [0.913,0.928,0.758,0.713,0.363,0.343,0.190,0.184,0.113,0.108,0.066,0.064,0.042,0.040,0.029,0.029,0.024,0.024,0.020,0.020];
A_d = A_d_cor   ;


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

% overwrite old method with new method
S_uu_n = S_uu_n_s_pol;

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

S_nn_e_wt = S_uu_n_wt./S_uu_t.*S_ftft_wt;

%% Calculate S_nn_e(j wd) 
% Retreive S_uu_n and S_uu_t for each jwt
S_uu_n_wd    = zeros(length(w_d), 1);
S_uu_d      = zeros(length(w_d), 1);
% S_ftft_wt = zeros(length(w_t), 1);
for i = 1:length(w_d)
    w_d_i = w_d(i);
    [d, ix] = min(abs(omega-w_d_i));
    S_uu_n_wd(i)     = S_uu_n(ix);
    S_uu_d(i)       = S_uu_mean(ix);
%     S_ftft_wt(i)    = S_ftft(ix);
end
%QUESTION: S_ftft = A_t? OR PERFORM SPECTRAL ANALYSIS ON fd SIGNAL?
%ANSWER: can use A_t as follows:
F_d = A_d/2*N;
S_fdfd_wd = (F_d.^2*dt/N)';

S_nn_e_wd = S_uu_n_wd./S_uu_d.*S_fdfd_wd;

%% Fitting model
xdata = [w_t; w_d];
ydata = [S_nn_e_wt'; S_nn_e_wd'];

fun = @(x,xdata)x(1)^2*(1+x(2)^2*xdata.^2).^-1; % functions shape

x0 = [0.5, 0.5]; % initial guess
lb = [0.01, 0.01]; % lower bound
ub = [Inf, Inf]; % upper bound
x = lsqcurvefit(fun,x0,xdata,ydata,lb,ub);


%% Plot remnant model fit

% figure
LineWidth = 1.3;
hold on
legend on
% plot(1,1)
plot(omega, fun(x, omega), 'DisplayName', 'model fit', 'LineWidth',LineWidth)
scatter(w_t, S_nn_e_wt, 'DisplayName', 'S_{nn_{e}}(j\omega_{t})', 'LineWidth',LineWidth)
scatter(w_d, S_nn_e_wd, 'DisplayName', 'S_{nn_{e}}(j\omega_{d})', 'LineWidth',LineWidth)


% more plotting settings
xlim([0.7*10^-1 10^2])
ylim([10^-5 10^0])

set(gca,'xscale','log') % set xscale to log
set(gca,'yscale','log') % set yscale to log

xlabel('\omega, rad/s')
ylabel('S_{nn_{e}} deg^{2} / Hz')

grid

% set ticks
Ticks = [10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 10^0];  
set(gca, 'YTickMode', 'manual', 'YTick', Ticks) 
set(gca, 'FontName', 'Times New Roman')

str = {'$$|H_{n}|^{2} = |0.19 \frac{1}{1+0.33 \cdot s} |^{2}$$'} ;
% str = 
% annotation('textbox','interpreter','latex','String',str,'FitBoxToText','on', 'EdgeColor', 'none');
% 
% hold off


%% Plot spectral analysis 
figure
hold on
legend on
legend('location', 'southwest')
LineWidth = 1.3;

name = append('S_{uu} individual runs (5x)');
p = plot(omega, S_uu_list, 'color', [0.8 0.8 0.8],'LineWidth',LineWidth); % 'DisplayName', name,
legend([p(1)], name, 'NumColumns',3);

% take average S_uu and plot
S_uu_mean = mean(S_uu_list, 2);
plot(omega, S_uu_mean, 'color', [0.1 0.1 0.1], 'DisplayName', 'S_{uu} five-run average','LineWidth',LineWidth);

%Plot TARGET magnitudes and frequencies
scatter(w_t, S_uu_t, 'DisplayName', 'S_{uu_{t}}, target frequency', 'MarkerEdgeColor', '#D95319','LineWidth',LineWidth) 
%Plot DISTURBANCE magnitude and frequencies
scatter(w_d, S_uu_d, 'DisplayName', 'S_{uu_{d}}, disturbance frequency', 'MarkerEdgeColor', '#EDB120','LineWidth',LineWidth)
%Plot interpolated S_uu_n values
scatter(omega, S_uu_n, 'marker','x','MarkerEdgeColor', '#0072BD', 'DisplayName', 'S_{uu_{n}}, interpolated ','LineWidth',LineWidth)

set(gca,'xscale','log') % set xscale to log
set(gca,'yscale','log') % set yscale to log
grid

xlabel('\omega, rad/s')
ylabel('S_{uu} deg^{2} / Hz')
% % set ticks
% Ticks = [10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 10^0];  
% set(gca, 'YTickMode', 'manual', 'YTick', Ticks) 

yticks ([10^-5 10^-4 10^-3 10^-2 10^-1 10^0 10^1]);
% font
set(gca, 'FontName', 'Times New Roman')

ylim([5*10^-4 2*10^2]);
xlim([0.7*10^-1 5*10^1]);