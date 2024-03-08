if true 
    data = HDRLOAD('data/B1_D3_02-April/ToTPitchTrackingData_Group1_Subject4_nr075_motOFF_fofuReal3.dat');

    % column descriptions:
    % 1:t, 2:ft, 3:fd, 5:e, 6:DYN u, 11:PCTRLS uy

    t = data(:,1);
    ft = data(:,2)*(180/pi);
    fd = data(:,3)*(180/pi);
    e = data(:, 5)*(180/pi);
    DYN_u = data(:, 6)*(180/pi);
    PCTRLS_uy = data(:, 11)*(180/pi);

    f_t = [t ft]';
    f_d = [t fd]';

    % Computing S_uu
    N = size(t,1);
    fs  = 100;   % sampling frequency
    dt  = 1/fs;  % sampling period
    T   = max(t); 
    omega = 2*pi*fs*(0:(N/2)-1)/N;

    F_DYN_u = dt*fft(DYN_u); % fourier transform of DYN_u

    S_uu = (1/T)*(F_DYN_u.*conj(F_DYN_u));

    %plot(omega, S_uu(1:round(N/2)-1))
    figure
    hold on
    legend('on')

    % multiple PSDs and averaging
    data = HDRLOAD('data/B1_D4_03-April/ToTPitchTrackingData_Group1_Subject1_nr098_motOFF_fofuReal3.dat');
    run3 = data(:, 6)*(180/pi); % u
    RUN3 = dt*fft(run3); % fourier transform
    S_run3 = (1/T)*(RUN3.*conj(RUN3));
    plot(omega, S_run3(1:round(N/2)-1), 'DisplayName', 'run3', 'color', [0.8 0.8 0.8])

    data = HDRLOAD('data/B1_D4_03-April/ToTPitchTrackingData_Group1_Subject1_nr099_motOFF_fofuReal5.dat');
    run4 = data(:, 6)*(180/pi); % u
    RUN4 = dt*fft(run4); % fourier transform
    S_run4 = (1/T)*(RUN4.*conj(RUN4));
    plot(omega, S_run4(1:round(N/2)-1), 'DisplayName', 'run4', 'color', [0.8 0.8 0.8])

    data = HDRLOAD('data/B1_D4_03-April/ToTPitchTrackingData_Group1_Subject1_nr100_motOFF_fofuReal1.dat');
    run5 = data(:, 6)*(180/pi); % u
    RUN5 = dt*fft(run5); % fourier transform
    S_run5 = (1/T)*(RUN5.*conj(RUN5));
    plot(omega, S_run5(1:round(N/2)-1), 'DisplayName', 'run5', 'color', [0.8 0.8 0.8])

    data = HDRLOAD('data/B1_D4_03-April/ToTPitchTrackingData_Group1_Subject1_nr101_motOFF_fofuReal4.dat');
    run6 = data(:, 6)*(180/pi); % u
    RUN6 = dt*fft(run6); % fourier transform
    S_run6 = (1/T)*(RUN6.*conj(RUN6));
    plot(omega, S_run6(1:round(N/2)-1), 'DisplayName', 'run6', 'color', [0.8 0.8 0.8])

    data = HDRLOAD('data/B1_D4_03-April/ToTPitchTrackingData_Group1_Subject1_nr102_motOFF_fofuReal2.dat');
    run7 = data(:, 6)*(180/pi); % u
    RUN7 = dt*fft(run7); % fourier transform
    S_run7 = (1/T)*(RUN7.*conj(RUN7));
    plot(omega, S_run7(1:round(N/2)-1), 'DisplayName', 'run7', 'color', [0.8 0.8 0.8])

    S_average = (S_run3 + S_run4 + S_run5 + S_run6 + S_run7)/5;
    plot(omega, S_run7(1:round(N/2)-1), 'DisplayName', 'Average', 'color', [0.1 0.1 0.1])

    % S_uu_n approx

    S_comb = [S_run3 S_run4 S_run5 S_run6 S_run7];
    S_uu_n = var(S_comb');
    scatter(omega, S_uu_n(1:round(N/2)-1), 'DisplayName', 'S_{uu_{n}}', 'marker','x','MarkerEdgeColor', 'b')



    % input signals
    T_ef    = 81.92;    % only last 81.92 seconds used
    w_m     = 2*pi / T_ef;    % measurement base frequency [rad/s]

    % Target signal:
    n_t     = [5 6 13 14 27 28 41 42 53 54 73 74 103 104 139 140 193 194 229 230]; % integer multiple
    w_t     = n_t.*w_m; % frequency of each sine [rad/s]
    A_t     = [0.51 0.49 0.34 0.33 0.16 0.15 0.08 0.08 0.06 0.05 ... % amplitude of each sine [deg]
               0.03 0.03 0.02 0.02 0.02 0.02 0.01 0.01 0.01 0.01]; 

    scatter(w_t, A_t, 'DisplayName', 'target frequency')

    % Disturbance signal:
    n_d     = [2 3 9 10 22 23 36 37 49 50 69 70 97 99 135 136 169 170 224 225]; % integer multiple
    w_d     = n_d.*w_m; % frequency of each sine [rad/s]
    A_d     = [0.10 0.15 0.30 0.30 0.19 0.18 0.12 0.12 0.12 0.12 ... % amplitude of each sine [deg]
               0.15 0.15 0.20 0.20 0.28 0.28 0.37 0.37 0.55 0.56];

    scatter(w_d, A_d, 'DisplayName', 'disturbance frequency')

    set(gca,'xscale','log')
    set(gca,'yscale','log')
end

%% --- fresh start ---
% close all
% clear all

data = HDRLOAD('data/B1_D3_02-April/ToTPitchTrackingData_Group1_Subject4_nr075_motOFF_fofuReal3.dat');

% column descriptions:
% 1:t, 2:ft, 3:fd, 5:e, 6:DYN u, 11:PCTRLS uy

t = data(:,1);
DYN_u = data(:, 6)*(180/pi);

% method 1: MATLAB docs
fs  = 100;   % sampling frequency
L = length(t);
f = 2*pi*fs*(0:(L/2))/L;

Y = dt*fft(DYN_u); % fourier transform 

P2 = abs(Y/L); % two sided spectrum
P1 = P2(1:L/2+1); % single sided spectrum
P1(2:end-1) = 2*P1(2:end-1);

loglog(f,P1, 'DisplayName', 'method 1') 
hold on

% method 2: from stochastic class -  DEZE GEBRUIKEN
N = length(t);
fs  = 100;   % sampling frequency
dt  = 1/fs;  % sampling period
T   = max(t); 
omega = 2*pi*fs*(0:(N/2)-1)/N;

F_DYN_u = dt*fft(DYN_u); % fourier transform of DYN_u

S_uu = (1/T)*(F_DYN_u.*conj(F_DYN_u));

loglog(omega, S_uu(1:round(N/2)-1), 'DisplayName', 'method 2')
