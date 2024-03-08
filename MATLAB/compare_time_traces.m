clear all

% enter filter specifications
group_list = [1]; % group 1 or group 2
subject_list = [10]; % what subjects to inspect
run_list = [90]; % what runs to inspect
motion_list = [0]; % motion off 0, motion on 1
fofu_list = [1:5]; % realisation of ft and fd

sim_data = HDRLOAD_list_sim(group_list, subject_list, run_list, motion_list, fofu_list);
pil_data = HDRLOAD_list(group_list, subject_list, run_list, motion_list, fofu_list);

figure
hold on
plot(sim_data{1}(:,1), sim_data{1}(:,11),  'DisplayName', 'input u - simulated pilot')
plot(pil_data{1}(:,1), pil_data{1}(:,11),  'DisplayName', 'input u - real pilot')
xlabel('time [sec]')
ylabel('u [rad]')
legend