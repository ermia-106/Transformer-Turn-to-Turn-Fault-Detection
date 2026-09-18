% ============================================================
% Generate_Dataset_Variable_Step.m
% تولید دیتاست با گام متغیر + بازنمونه‌برداری به گام ۰.۰۰۰۵
% ============================================================
clc; clear; close all;

model_name = 'transformer_fault_model'; 
open_system(model_name);

fprintf('========================================================\n');
fprintf('   تولید دیتاست با گام متغیر + بازنمونه‌برداری\n');
fprintf('========================================================\n\n');

% -------------------------------------------------------------------------
% ۱. پارامترها
% -------------------------------------------------------------------------
V_pri = 20000;
V_sec = 400;

load_powers = [5e6, 6e6, 8e6, 10e6, 12e6];
fault_percents = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20];
fault_times = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18];

% -------------------------------------------------------------------------
% ۲. تنظیمات شبیه‌سازی (گام متغیر)
% -------------------------------------------------------------------------
sim_stop_time = 0.2;
target_sample_time = 0.0005;  % گام هدف برای بازنمونه‌برداری

% تنظیم پارامترهای سیمولینک برای گام متغیر
set_param(model_name, 'StopTime', num2str(sim_stop_time));
set_param(model_name, 'Solver', 'ode23tb');        % مناسب برای سیستم‌های سخت
set_param(model_name, 'SolverType', 'Variable-step');
set_param(model_name, 'MaxStep', '0.0002');        % حداکثر گام (برای دقت)
set_param(model_name, 'MinStep', '1e-8');          % حداقل گام
set_param(model_name, 'InitialStep', '1e-6');      % گام اولیه
set_param(model_name, 'RelTol', '1e-4');           % تولرانس نسبی
set_param(model_name, 'AbsTol', '1e-6');           % تولرانس مطلق

% solver های جایگزین در صورت نیاز:
% 'ode15s' (سیستم‌های سخت)، 'ode23s' (سخت با گام بزرگتر)، 'ode23t' (نوسانی)

fprintf('گام هدف برای بازنمونه‌برداری: %.4f ثانیه (%.1f کیلوهرتز)\n', ...
    target_sample_time, 1/target_sample_time/1000);
fprintf('زمان کل شبیه‌سازی: %.2f ثانیه\n', sim_stop_time);
fprintf('تعداد نمونه هدف در هر سناریو: %d\n\n', round(sim_stop_time / target_sample_time));

% -------------------------------------------------------------------------
% ۳. تعداد سناریوها
% -------------------------------------------------------------------------
total_scenarios = 150;
num_fault = 90;
num_normal = 60;

fprintf('تعداد کل سناریوها: %d\n', total_scenarios);
fprintf('   سناریوهای سالم: %d (۴۰%%)\n', num_normal);
fprintf('   سناریوهای خطا: %d (۶۰%%)\n\n', num_fault);

% -------------------------------------------------------------------------
% ۴. تولید سناریوهای سالم (۶۰ مورد)
% -------------------------------------------------------------------------
fprintf('--- تولید سناریوهای سالم (%d مورد) ---\n', num_normal);

dataset_all = [];
scenario_counter = 0;
normal_repeats = [12, 12, 12, 12, 12];

% بردار زمان هدف (گام ثابت)
target_time = (0:target_sample_time:sim_stop_time)';
target_time = target_time(1:end-1);  % حذف آخرین نمونه

rng(2026);

for L = 1:length(load_powers)
    P = load_powers(L);
    for rep = 1:normal_repeats(L)
        scenario_counter = scenario_counter + 1;
        
        assignin('base', 'P_load', P);
        assignin('base', 'T_fault', 999);
        assignin('base', 'V_A2', V_pri * 0.05);
        assignin('base', 'V_A2_sec', V_sec * 0.05);
        
        % شبیه‌سازی با گام متغیر
        simOut = sim(model_name);
        [I_abc, time_vec] = extract_signals(simOut);
        
        % بازنمونه‌برداری با درون‌یابی اسپلاین
        Ia_interp = interp1(time_vec, I_abc(:,1), target_time, 'spline');
        Ib_interp = interp1(time_vec, I_abc(:,2), target_time, 'spline');
        Ic_interp = interp1(time_vec, I_abc(:,3), target_time, 'spline');
        
        N = length(target_time);
        run_data = [target_time, Ia_interp, Ib_interp, Ic_interp, ...
                    repmat(P, N, 1), repmat(0, N, 1), ...
                    repmat(scenario_counter, N, 1), repmat(0, N, 1)];
        dataset_all = [dataset_all; run_data];
        
        fprintf('سناریو %d/%d [سالم] توان: %.1f MW | نمونه‌ها: %d\n', ...
            scenario_counter, total_scenarios, P/1e6, N);
    end
end

% -------------------------------------------------------------------------
% ۵. تولید سناریوهای خطا (۹۰ مورد)
% -------------------------------------------------------------------------
fprintf('\n--- تولید سناریوهای خطا (%d مورد) ---\n', num_fault);

all_combos = [];
for L = 1:length(load_powers)
    for p = 1:length(fault_percents)
        for t = 1:length(fault_times)
            all_combos = [all_combos; load_powers(L), fault_percents(p), fault_times(t)];
        end
    end
end

selected_indices = randperm(size(all_combos, 1), num_fault);
selected_combos = all_combos(selected_indices, :);

unique_combos = unique(selected_combos, 'rows');
while size(unique_combos, 1) < num_fault
    extra_indices = randperm(size(all_combos, 1), num_fault - size(unique_combos, 1));
    extra_combos = all_combos(extra_indices, :);
    selected_combos = [unique_combos; extra_combos];
    unique_combos = unique(selected_combos, 'rows');
end
selected_combos = unique_combos(1:num_fault, :);

for i = 1:num_fault
    P = selected_combos(i, 1);
    pct = selected_combos(i, 2);
    tf = selected_combos(i, 3);
    
    scenario_counter = scenario_counter + 1;
    
    assignin('base', 'P_load', P);
    assignin('base', 'V_A2', V_pri * pct);
    assignin('base', 'V_A2_sec', V_sec * pct);
    assignin('base', 'T_fault', tf);
    
    simOut = sim(model_name);
    [I_abc, time_vec] = extract_signals(simOut);
    
    % بازنمونه‌برداری با درون‌یابی اسپلاین
    Ia_interp = interp1(time_vec, I_abc(:,1), target_time, 'spline');
    Ib_interp = interp1(time_vec, I_abc(:,2), target_time, 'spline');
    Ic_interp = interp1(time_vec, I_abc(:,3), target_time, 'spline');
    
    N = length(target_time);
    run_data = [target_time, Ia_interp, Ib_interp, Ic_interp, ...
                repmat(P, N, 1), repmat(pct, N, 1), ...
                repmat(scenario_counter, N, 1), repmat(1, N, 1)];
    dataset_all = [dataset_all; run_data];
    
    fprintf('سناریو %d/%d [خطا] توان: %.1f MW | خطا: %.0f%% | زمان: %.3fs | نمونه‌ها: %d\n', ...
        scenario_counter, total_scenarios, P/1e6, pct*100, tf, N);
end

% -------------------------------------------------------------------------
% ۶. ذخیره دیتاست
% -------------------------------------------------------------------------
colnames = {'Time', 'Ia', 'Ib', 'Ic', 'Load_Power', 'Fault_Percent', 'Scenario_ID', 'Label'};
T_raw = array2table(dataset_all, 'VariableNames', colnames);
writetable(T_raw, 'ML_Transformer_FixedStep_2kHz.csv');
save('ML_Transformer_FixedStep_2kHz.mat', 'dataset_all');

fprintf('\n========================================================\n');
fprintf('✅ دیتاست خام با گام ثابت در ML_Transformer_FixedStep_2kHz.csv ذخیره شد.\n');
fprintf('   تعداد کل سناریوها: %d\n', scenario_counter);
fprintf('   سناریوهای سالم: %d (%.1f%%)\n', num_normal, num_normal/total_scenarios*100);
fprintf('   سناریوهای خطا: %d (%.1f%%)\n', num_fault, num_fault/total_scenarios*100);
fprintf('   گام هدف: %.4f ثانیه (%.1f کیلوهرتز)\n', target_sample_time, 1/target_sample_time/1000);
fprintf('   تعداد نمونه در هر سناریو: %d\n', length(target_time));
fprintf('========================================================\n');

% -------------------------------------------------------------------------
% تابع کمکی استخراج سیگنال
% -------------------------------------------------------------------------
function [I_abc, time_vec] = extract_signals(simOut)
    var_names = simOut.who;
    data_obj = simOut.get(var_names{1});
    
    if isstruct(data_obj)
        I_abc = data_obj.signals.values;
        time_vec = data_obj.time;
    elseif isa(data_obj, 'Simulink.SimulationData.Dataset')
        ds = data_obj.getElement(1);
        I_abc = ds.Values.Data;
        time_vec = ds.Values.Time;
    else
        I_abc = data_obj(:, 2:end);
        time_vec = data_obj(:, 1);
    end
end