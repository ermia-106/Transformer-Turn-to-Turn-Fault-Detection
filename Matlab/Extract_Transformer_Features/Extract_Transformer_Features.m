% ============================================================
% Extract_All_Features_FixedStep.m
% استخراج ۱۹ ویژگی زمان + ۳۰ ویژگی موجک = ۴۹ ویژگی
% از دیتاست با گام ثابت (ML_Transformer_FixedStep_2kHz.csv)
% ============================================================
clc; clear; close all;

fprintf('========================================================\n');
fprintf('   استخراج همه ویژگی‌های زمان و موجک (۴۹ ویژگی)\n');
fprintf('   از دیتاست با گام ثابت (۲ کیلوهرتز)\n');
fprintf('========================================================\n\n');

% --------------------- ۱. بارگذاری دیتاست خام ---------------------
try
    opts = detectImportOptions('ML_Transformer_FixedStep_2kHz.csv');
    raw = readtable('ML_Transformer_FixedStep_2kHz.csv', opts);
    fprintf('✅ دیتاست با گام ثابت (ML_Transformer_FixedStep_2kHz.csv) بارگذاری شد.\n');
catch
    try
        opts = detectImportOptions('ML_Transformer_FixedStep.csv');
        raw = readtable('ML_Transformer_FixedStep.csv', opts);
        fprintf('✅ دیتاست با گام ثابت (ML_Transformer_FixedStep.csv) بارگذاری شد.\n');
    catch
        error('❌ فایل دیتاست با گام ثابت یافت نشد!');
    end
end

fprintf('تعداد سناریوها: %d\n', length(unique(raw.Scenario_ID)));
fprintf('تعداد کل رکوردها: %d\n', height(raw));
fprintf('گام زمانی: %.4f ثانیه\n', mean(diff(raw.Time(1:10))));

% --------------------- ۲. تنظیمات پنجره‌بندی و موجک ---------------------
window_size = 80 ;      % طول هر پنجره (نمونه)
step_size = 40;        % گام حرکت (۵۰٪ همپوشانی)
wavelet_name = 'db4';  % موجک مادر Daubechies 4
level = 5;             % سطح تجزیه

scenario_ids = unique(raw.Scenario_ID);
num_scenarios = length(scenario_ids);

feature_matrix = [];
fprintf('\nشروع پردازش %d سناریو...\n', num_scenarios);

% --------------------- ۳. حلقه اصلی روی سناریوها ---------------------
for i = 1:num_scenarios
    sc_id = scenario_ids(i);
    sc_data = raw(raw.Scenario_ID == sc_id, :);
    
    Ia = sc_data.Ia;
    Ib = sc_data.Ib;
    Ic = sc_data.Ic;
    label_full = sc_data.Label;
    load_p = sc_data.Load_Power(1);
    fault_p = sc_data.Fault_Percent(1);
    
    num_samples = length(Ia);
    
    for w_start = 1:step_size:(num_samples - window_size + 1)
        w_end = w_start + window_size - 1;
        
        % ---- برچسب‌گذاری پنجره (حذف پنجره‌های مرزی) ----
        fault_ratio = mean(label_full(w_start:w_end));
        if fault_ratio > 0.1 && fault_ratio < 0.9
            continue;
        end
        w_label = round(fault_ratio);
        
        % ---- برش پنجره ----
        Ia_win = Ia(w_start:w_end);
        Ib_win = Ib(w_start:w_end);
        Ic_win = Ic(w_start:w_end);
        
        % ============================================================
        % بخش اول: ۱۹ ویژگی حوزه زمان
        % ============================================================
        
        % ۱. RMS (۳)
        rms_a = rms(Ia_win);
        rms_b = rms(Ib_win);
        rms_c = rms(Ic_win);
        
        % ۲. پیک (۳)
        peak_a = max(abs(Ia_win));
        peak_b = max(abs(Ib_win));
        peak_c = max(abs(Ic_win));
        
        % ۳. انحراف معیار (۳)
        std_a = std(Ia_win);
        std_b = std(Ib_win);
        std_c = std(Ic_win);
        
        % ۴. ضریب قله (۳)
        crest_a = peak_a / (rms_a + eps);
        crest_b = peak_b / (rms_b + eps);
        crest_c = peak_c / (rms_c + eps);
        
        % ۵. THD (۳)
        Ia_fft = fft(Ia_win);
        Ib_fft = fft(Ib_win);
        Ic_fft = fft(Ic_win);
        thd_a = norm(abs(Ia_fft(3:end))) / (abs(Ia_fft(2)) + eps);
        thd_b = norm(abs(Ib_fft(3:end))) / (abs(Ib_fft(2)) + eps);
        thd_c = norm(abs(Ic_fft(3:end))) / (abs(Ic_fft(2)) + eps);
        
        % ۶. مؤلفه‌های متقارن (۳)
        I_A = Ia_fft(2);
        I_B = Ib_fft(2);
        I_C = Ic_fft(2);
        a = exp(1i * 2 * pi / 3);
        I0 = abs((I_A + I_B + I_C) / 3);
        I1 = abs((I_A + a*I_B + (a^2)*I_C) / 3);
        I2 = abs((I_A + (a^2)*I_B + a*I_C) / 3);
        
        % ۷. نسبت نامتعادلی (۱)
        unbalance = I2 / (I1 + eps);
        
        % ============================================================
        % بخش دوم: ۳۰ ویژگی حوزه موجک
        % ============================================================
        
        % ---- تجزیه فاز A ----
        [C_a, L_a] = wavedec(Ia_win, level, wavelet_name);
        D1_a = detcoef(C_a, L_a, 1);
        D2_a = detcoef(C_a, L_a, 2);
        D3_a = detcoef(C_a, L_a, 3);
        D4_a = detcoef(C_a, L_a, 4);
        D5_a = detcoef(C_a, L_a, 5);
        
        % ---- تجزیه فاز B ----
        [C_b, L_b] = wavedec(Ib_win, level, wavelet_name);
        D1_b = detcoef(C_b, L_b, 1);
        D2_b = detcoef(C_b, L_b, 2);
        D3_b = detcoef(C_b, L_b, 3);
        D4_b = detcoef(C_b, L_b, 4);
        D5_b = detcoef(C_b, L_b, 5);
        
        % ---- تجزیه فاز C ----
        [C_c, L_c] = wavedec(Ic_win, level, wavelet_name);
        D1_c = detcoef(C_c, L_c, 1);
        D2_c = detcoef(C_c, L_c, 2);
        D3_c = detcoef(C_c, L_c, 3);
        D4_c = detcoef(C_c, L_c, 4);
        D5_c = detcoef(C_c, L_c, 5);
        
        % ---- انرژی (۱۵) ----
        energy_D1_a = sum(D1_a.^2);
        energy_D2_a = sum(D2_a.^2);
        energy_D3_a = sum(D3_a.^2);
        energy_D4_a = sum(D4_a.^2);
        energy_D5_a = sum(D5_a.^2);
        
        energy_D1_b = sum(D1_b.^2);
        energy_D2_b = sum(D2_b.^2);
        energy_D3_b = sum(D3_b.^2);
        energy_D4_b = sum(D4_b.^2);
        energy_D5_b = sum(D5_b.^2);
        
        energy_D1_c = sum(D1_c.^2);
        energy_D2_c = sum(D2_c.^2);
        energy_D3_c = sum(D3_c.^2);
        energy_D4_c = sum(D4_c.^2);
        energy_D5_c = sum(D5_c.^2);
        
        % ---- آنتروپی (۵، فاز A) ----
        p1 = (D1_a.^2) / (energy_D1_a + eps);
        p2 = (D2_a.^2) / (energy_D2_a + eps);
        p3 = (D3_a.^2) / (energy_D3_a + eps);
        p4 = (D4_a.^2) / (energy_D4_a + eps);
        p5 = (D5_a.^2) / (energy_D5_a + eps);
        
        entropy_D1_a = -sum(p1 .* log2(p1 + eps));
        entropy_D2_a = -sum(p2 .* log2(p2 + eps));
        entropy_D3_a = -sum(p3 .* log2(p3 + eps));
        entropy_D4_a = -sum(p4 .* log2(p4 + eps));
        entropy_D5_a = -sum(p5 .* log2(p5 + eps));
        
        % ---- واریانس (۵، فاز A) ----
        var_D1_a = var(D1_a);
        var_D2_a = var(D2_a);
        var_D3_a = var(D3_a);
        var_D4_a = var(D4_a);
        var_D5_a = var(D5_a);
        
        % ---- انحراف معیار (۵، فاز A) ----
        std_D1_a = std(D1_a);
        std_D2_a = std(D2_a);
        std_D3_a = std(D3_a);
        std_D4_a = std(D4_a);
        std_D5_a = std(D5_a);
        
        % ============================================================
        % ساخت بردار نهایی: ۱۹ زمان + ۳۰ موجک = ۴۹ ویژگی + ۴ کمکی
        % ============================================================
        time_features = [rms_a, rms_b, rms_c, ...
                         peak_a, peak_b, peak_c, ...
                         std_a, std_b, std_c, ...
                         crest_a, crest_b, crest_c, ...
                         thd_a, thd_b, thd_c, ...
                         I0, I1, I2, unbalance];
        
        energy_features = [energy_D1_a, energy_D2_a, energy_D3_a, energy_D4_a, energy_D5_a, ...
                           energy_D1_b, energy_D2_b, energy_D3_b, energy_D4_b, energy_D5_b, ...
                           energy_D1_c, energy_D2_c, energy_D3_c, energy_D4_c, energy_D5_c];
        
        entropy_features = [entropy_D1_a, entropy_D2_a, entropy_D3_a, entropy_D4_a, entropy_D5_a];
        
        var_features = [var_D1_a, var_D2_a, var_D3_a, var_D4_a, var_D5_a];
        
        std_features = [std_D1_a, std_D2_a, std_D3_a, std_D4_a, std_D5_a];
        
        meta_features = [load_p, fault_p, sc_id, w_label];
        
        row = [time_features, energy_features, entropy_features, var_features, std_features, meta_features];
        
        % بررسی تعداد ستون‌ها
        if length(row) ~= 53
            fprintf('⚠️ خطا در سناریو %d: تعداد ستون‌ها %d است (باید ۵۳ باشد)\n', sc_id, length(row));
            continue;
        end
        
        feature_matrix = [feature_matrix; row];
    end
    
    if mod(i, 10) == 0
        fprintf('پردازش %d/%d سناریو...\n', i, num_scenarios);
    end
end

% ============================================================
% ۴. نام‌گذاری ستون‌ها
% ============================================================

% ۱۹ ویژگی زمان
time_names = {
    'RMS_Ia','RMS_Ib','RMS_Ic', ...
    'Peak_Ia','Peak_Ib','Peak_Ic', ...
    'Std_Ia','Std_Ib','Std_Ic', ...
    'Crest_Ia','Crest_Ib','Crest_Ic', ...
    'THD_Ia','THD_Ib','THD_Ic', ...
    'I0_Zero','I1_Pos','I2_Neg','Unbalance'
};

% ۳۰ ویژگی موجک
wavelet_names = {};

% انرژی (۱۵)
for phase = {'a','b','c'}
    for lvl = 1:5
        wavelet_names{end+1} = sprintf('Energy_%s_D%d', phase{1}, lvl);
    end
end

% آنتروپی (۵)
for lvl = 1:5
    wavelet_names{end+1} = sprintf('Entropy_D%d_a', lvl);
end

% واریانس (۵)
for lvl = 1:5
    wavelet_names{end+1} = sprintf('Var_D%d_a', lvl);
end

% انحراف معیار (۵)
for lvl = 1:5
    wavelet_names{end+1} = sprintf('Std_D%d_a', lvl);
end

% ۴ ستون کمکی
meta_names = {'Load_Power','Fault_Percent','Scenario_ID','Label'};

% ترکیب همه
all_names = [time_names, wavelet_names, meta_names];

% ============================================================
% ۵. ذخیره در CSV
% ============================================================
if size(feature_matrix, 2) == length(all_names)
    T_feat = array2table(feature_matrix, 'VariableNames', all_names);
    writetable(T_feat, 'ML_Features_All_49_new6.csv');
    
    fprintf('\n========================================================\n');
    fprintf('✅ استخراج ۴۹ ویژگی (۱۹ زمان + ۳۰ موجک) با موفقیت کامل شد!\n');
    fprintf('   تعداد کل نمونه‌ها: %d\n', size(feature_matrix, 1));
    fprintf('   تعداد ویژگی‌ها: %d\n', size(feature_matrix, 2)-4);
    fprintf('   سالم: %d | خطا: %d\n', sum(T_feat.Label==0), sum(T_feat.Label==1));
    fprintf('   درصد خطا: %.2f%%\n', sum(T_feat.Label==1)/height(T_feat)*100);
    fprintf('📁 فایل خروجی: ML_Features_All_49.csv\n');
    fprintf('========================================================\n');
else
    fprintf('❌ خطا: تعداد ستون‌ها با نام‌ها مطابقت ندارد!\n');
    fprintf('   ستون‌ها: %d | نام‌ها: %d\n', size(feature_matrix, 2), length(all_names));
end