K_list_start = [];
T_list_start = [];

K_list_end = [];
T_list_end = [];

for subject=1:13
%     sub_str = append('Subject_', string(subject));
    
    run = [6:10];
    analyse_actual_data 
    K = x(1);
    K_list_start = [K_list_start K];

    T_l = x(2);
    T_list_start = [T_list_start T_l];
    
%     rem_par.K.(sub_str)(1) = x(1);
%     rem_par.T.(sub_str)(1) = x(2);
    
    run = [96:100];
    analyse_actual_data 
    K = x(1);
    K_list_end = [K_list_end K];

    T_l = x(2);
    T_list_end = [T_list_end T_l];
    
%     rem_par.K.(sub_str)(2) = x(1);
%     rem_par.T.(sub_str)(2) = x(2);    

end

figure
boxplot([K_list_start', K_list_end'],'Labels',{'First 5 runs','Last 5 runs'})
ylabel('K, -')

figure
boxplot([T_list_start', T_list_end'],'Labels',{'First 5 runs','Last 5 runs'})
ylabel('T_{l}, s')

