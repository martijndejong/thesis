load Parameters/Results.mat
load Parameters/new_remnant5_parameters.mat
parameters = Data;


%% User input
% set group and phase to be simulated
group       = 'NM';         %'NM'/'M'
phase       = 'training';   %'training'/'test'


%% Simulation settings based on user input
% Enable or disable motion reponse loop based on settings
if (strcmp(group,'M') && strcmp(phase,'training')) || (strcmp(group,'NM') && strcmp(phase,'test'))
    Mot_Resp = true; % Motion response is enabled in simulink pilot model
else
    Mot_Resp = false; % Motion response is disabled in simulink pilot model
end

% empty lists for unskilled parameters
w_nm_unsk_list = [];
z_nm_unsk_list = [];
K_v_unsk_list = [];
T_lead_unsk_list = [];
T_lag_unsk_list = [];
t_v_unsk_list = [];
K_n_unsk_list = [];
T_l_unsk_list = [];

% empty lists for skilled parameters
w_nm_sk_list = [];
z_nm_sk_list = [];
K_v_sk_list = [];
T_lead_sk_list = [];
T_lag_sk_list = [];
t_v_sk_list = [];
K_n_sk_list = [];
T_l_sk_list = [];

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
    
    
    % add unskilled parameters to lists
    w_nm_unsk_list = [w_nm_unsk_list; w_nm_list(1:20)];
    z_nm_unsk_list = [z_nm_unsk_list; z_nm_list(1:20)];
    K_v_unsk_list = [K_v_unsk_list; K_v_list(1:20)];
    T_lead_unsk_list = [T_lead_unsk_list; T_lead_list(1:20)];
    T_lag_unsk_list = [T_lag_unsk_list; T_lag_list(1:20)];
    t_v_unsk_list = [t_v_unsk_list; t_v_list(1:20)];
    K_n_unsk_list = [K_n_unsk_list; K_n_list(1:20)];
    T_l_unsk_list = [T_l_unsk_list; T_l_list(1:20)];
    
    % add skilled parameters to lists
    w_nm_sk_list = [w_nm_sk_list; w_nm_list(81:100)];
    z_nm_sk_list = [z_nm_sk_list; z_nm_list(81:100)];
    K_v_sk_list = [K_v_sk_list; K_v_list(81:100)];
    T_lead_sk_list = [T_lead_sk_list; T_lead_list(81:100)];
    T_lag_sk_list = [T_lag_sk_list; T_lag_list(81:100)];
    t_v_sk_list = [t_v_sk_list; t_v_list(81:100)];
    K_n_sk_list = [K_n_sk_list; K_n_list(81:100)];
    T_l_sk_list = [T_l_sk_list; T_l_list(81:100)];
end


% figure
set(gcf,'Position',[100 0 750 900])
plot1 = subplot(4,2,1, 'align');
boxplot([w_nm_unsk_list, w_nm_sk_list],'Labels',{'Unskilled','Skilled'})
ylabel('\omega_{nm}, rad/s')
ylim([-1, 62]) % two outliers for first 15 runs: 8875 and 17525
% x = [0.6,0.5];
% y = [mean(yLimits), yLimits(2)];
str = "Outliers at 8875,"+newline+"17525";
annotation('textarrow',[0.262,0.22],[0.85,0.92], 'FontSize', 8,...
                               'FontName', 'Times New Roman',...
                               'String',str,...
                               'HeadWidth', 6, 'HeadLength', 6,...
                               'HorizontalAlignment', 'right')
set(gca, 'FontName', 'Times New Roman')
grid

% figure
plot2 = subplot(4,2,2, 'align');
boxplot([z_nm_unsk_list, z_nm_sk_list],'Labels',{'Unskilled','Skilled'})
ylabel('\zeta_{nm}, -')
ylim([0, 4.5]) % three outliers for first 15 runs: 24, 285, and 5171
set(gca, 'FontName', 'Times New Roman')
str = "Outliers at 24,"+newline+" 285,"+newline+" 5171";
annotation('textarrow',[0.7,0.67],[0.85,0.92], 'FontSize', 8,...
                               'FontName', 'Times New Roman',...
                               'String',str,...
                               'HeadWidth', 6, 'HeadLength', 6,...
                           'HorizontalAlignment', 'right')
plot2.Position = [plot2.Position(1), plot1.Position(2), plot2.Position(3), plot2.Position(4)];                     
set(gca, 'FontName', 'Times New Roman')
grid

% figure
plot3 = subplot(4,2,3, 'align');
boxplot([K_v_unsk_list, K_v_sk_list],'Labels',{'Unskilled','Skilled'})
ylabel('K_{v}, -')
ylim([0, 6.8]) % no outliers
set(gca, 'FontName', 'Times New Roman')
grid

% figure
plot4 = subplot(4,2,4, 'align');
%[x_0 y_0 width heigth]
boxplot([t_v_unsk_list, t_v_sk_list],'Labels',{'Unskilled','Skilled'})
ylabel('t_{v}, s')
ylim([0, 0.85]) % one outlier at 2.37 first 15 runs
str = "Outlier at 2.37"; 
annotation('textarrow',[0.712,0.67],[0.63,0.7], 'FontSize', 8,...
                               'FontName', 'Times New Roman',...
                               'String',str,...
                               'HeadWidth', 6, 'HeadLength', 6,...
                           'HorizontalAlignment', 'right')
plot4.Position = [plot2.Position(1), plot4.Position(2), plot2.Position(3), plot2.Position(4)];
set(gca, 'FontName', 'Times New Roman')
grid

% figure
plot5 = subplot(4,2,5, 'align');
boxplot([T_lead_unsk_list, T_lead_sk_list],'Labels',{'Unskilled','Skilled'})
ylabel('T_{lead}, s')
ylim([0, 2.4]) % no outliers
set(gca, 'FontName', 'Times New Roman')
grid

% figure
plot6 = subplot(4,2,6, 'align');
boxplot([T_lag_unsk_list, T_lag_sk_list],'Labels',{'Unskilled','Skilled'})
ylabel('T_{lag}, s')
ylim([0, 11.9]) % no outliers
plot6.Position = [plot2.Position(1), plot6.Position(2), plot2.Position(3), plot2.Position(4)];
set(gca, 'FontName', 'Times New Roman')
grid


% figure
plot7 = subplot(4,2,7, 'align');
boxplot([K_n_unsk_list, K_n_sk_list],'Labels',{'Unskilled','Skilled'})
ylabel('K_{n}, -')
ylim([0, 3.3]) % one outlier 47 last 15 runs
str = "Outlier at 47"; 
annotation('textarrow',[0.8,0.82],[0.22,0.26], 'FontSize', 8,...
                               'FontName', 'Times New Roman',...
                               'String',str,...
                               'HeadWidth', 6, 'HeadLength', 6,...
                           'HorizontalAlignment', 'right')
set(gca, 'FontName', 'Times New Roman')
grid

% figure
plot8 = subplot(4,2,8, 'align');
boxplot([T_l_unsk_list, T_l_sk_list],'Labels',{'Unskilled','Skilled'})
ylabel('T_{n,lag}, s')
ylim([0, 3])% one outlier at 184 last 15 runs
set(gca, 'FontName', 'Times New Roman')
str = "Outlier at 184"; 
annotation('textarrow',[0.34,0.37],[0.22,0.26], 'FontSize', 8,...
                               'FontName', 'Times New Roman',...
                               'String',str,...
                               'HeadWidth', 6, 'HeadLength', 6,...
                           'HorizontalAlignment', 'right')
plot8.Position = [plot2.Position(1), plot8.Position(2), plot2.Position(3), plot8.Position(4)];
grid

%[x_0 y_0 width heigth]


