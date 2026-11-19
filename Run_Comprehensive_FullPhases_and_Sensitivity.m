%% ========================================================================
%  File: Run_Comprehensive_FullPhases_and_Sensitivity.m
%  Description: 全周期时相 (T00-T90) 与超参数敏感性网格搜索 (内存秒速版)
%  Outputs: Full-Cycle Evaluation Table & Figure S1 (Alpha Sensitivity Curve)
% =========================================================================
clear; clc; close all;

% ------------------------- 路径配置 -------------------------------------
main_path   = 'E:\D盘数据转移\data四维剂量重建\DIR-Lab\';
result_path = 'D:\0临床科研\四维剂量重建(给喜雪改投版本)\result\R2\';
if ~exist(result_path, 'dir'), mkdir(result_path); end
[alpha_kide,kappa] = R2_LoadParameters(result_path);

%% ========================================================================
%  PART 1: 全周期呼吸时相 (T00-T90) 预测验证 (回答 Reviewer #2 Q6)
% ========================================================================
fprintf('====================================================================\n');
fprintf('PART 1: Descriptive full-cycle analysis (Phases T00 to T90 across 10 Cases)\n');
fprintf('====================================================================\n');

all_phase_rms = zeros(10, 10); % [10 cases x 10 phases]
all_phase_jac = zeros(10, 10);

% 预缓存结构体
all_phase_rms_noisy = nan(10,10);
all_phase_jac_noisy = nan(10,10);

for case_id = 1:10
    fprintf('  [预缓存与全时相计算] Case %02d / 10 ...\n', case_id);
    case_path = fullfile(main_path, sprintf('Case%dPack', case_id));
    processed_file = fullfile(case_path, 'Processed', sprintf('Case%d_DVF_Depth.mat', case_id));
    assert(isfile(processed_file),'Missing cache %s',processed_file);
    
    data = load(processed_file, 'depth_maps', 'DVFs', 'pts_T00', 'pts_T50');
    [img_size, vs] = get_dirlab_params(case_id);
    
    % 严格使用 Phase 1-5 提取先验
    train_depths = reshape(data.depth_maps(:,:,1:5), [], 5)';
    mean_depth = mean(train_depths, 1);
    [coeff_S, score_S, ~] = pca(train_depths, 'Economy', true);
    Z_train = score_S(:, 1:3);
    
    if prod(img_size) > 1e6, scale_factor = 0.5; else, scale_factor = 1.0; end
    small_size = max(floor(img_size * scale_factor), [1, 1, 1]);
    
    V_train = zeros(5, prod(small_size)*3, 'single');
    for i = 1:5
        dvf_s = zeros([small_size, 3], 'single');
        for c = 1:3, dvf_s(:,:,:,c) = imresize3(data.DVFs{i}(:,:,:,c), small_size, 'linear'); end
        V_train(i, :) = dvf_s(:);
    end
    mean_V = mean(V_train, 1);
    [coeff_V, score_V, latent_V] = pca(V_train, 'Economy', true);
    Y_train = score_V(1:5, 1:3);
    
    H = Z_train' * Z_train; scale_H = max(abs(H(:))); if scale_H < 1e-6, scale_H = 1; end
    
    % 训练集交叉验证软谱参数
    % R2: kappa selected using training-only CV.
    max_L = max(latent_V(1:3));
    penalty_diag = (max_L - latent_V(1:3)) ./ (latent_V(1:3) + kappa * max_L + 1e-6);
    penalty_diag(1) = 0;
    % R2: alpha selected using training-only CV.
    W_kide = (H + alpha_kide * scale_H * diag(penalty_diag)) \ (Z_train' * Y_train);
    
    % DVF components are in original voxel units.
    
    % 评估全周期 10 个时相 (T00 到 T90)
    for p = 1:10
        curr_d = data.depth_maps(:,:,p);
        z_curr = (curr_d(:)' - mean_depth) * coeff_S;
        y_pred = z_curr(1:3) * W_kide;
        pred_dvf_s = reshape(mean_V + y_pred * coeff_V(:, 1:3)', [small_size, 3]);
        
        gt_dvf_s = zeros([small_size, 3], 'single');
        for c = 1:3, gt_dvf_s(:,:,:,c) = imresize3(data.DVFs{p}(:,:,:,c), small_size, 'linear'); end
        
        diff_vec = (pred_dvf_s - gt_dvf_s) .* reshape(vs, [1, 1, 1, 3]);
        all_phase_rms(case_id, p) = sqrt(mean(sum(diff_vec.^2, 4), 'all'));
        all_phase_jac(case_id, p) = R2_JacobianFolding(pred_dvf_s,vs,img_size);
        if p==6, seed=case_id+100; else, seed=2000+10*case_id+p; end
        noisy=R2_PerturbDepth(curr_d,vs,seed);
        z_noisy=(noisy(:)'-mean_depth)*coeff_S(:,1:3);
        pred_noisy=reshape(mean_V+(z_noisy*W_kide)*coeff_V(:,1:3)',[small_size,3]);
        diff_noisy=(pred_noisy-gt_dvf_s).*reshape(vs,[1,1,1,3]);
        all_phase_rms_noisy(case_id,p)=sqrt(mean(sum(diff_noisy.^2,4),'all'));
        all_phase_jac_noisy(case_id,p)=R2_JacobianFolding(pred_noisy,vs,img_size);
    end
    clear data V_train coeff_V;
end

phases_str = {'T00', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90'};
% Table 2: descriptive case-level phase summaries (NOT independent volumes).
status=["Reference (fit)";repmat("Training",4,1);repmat("Held-out phase",5,1)];
T=table(string(phases_str(:)),status,mean(all_phase_rms,1)',std(all_phase_rms,0,1)',mean(all_phase_jac,1)', ...
    mean(all_phase_rms_noisy,1)',std(all_phase_rms_noisy,0,1)',mean(all_phase_jac_noisy,1)', ...
    'VariableNames',{'Phase','Status','Clean_RMS_mean','Clean_RMS_SD','Clean_Fold_mean_pct','Noisy_RMS_mean','Noisy_RMS_SD','Noisy_Fold_mean_pct'});
disp(T); writetable(T,fullfile(result_path,'R2_FullPhases.csv'));
% Within-patient average over the five held-out phases, then across cases.
H=table([mean(mean(all_phase_rms(:,6:10),2));mean(mean(all_phase_rms_noisy(:,6:10),2))], ...
    [std(mean(all_phase_rms(:,6:10),2));std(mean(all_phase_rms_noisy(:,6:10),2))], ...
    [mean(all_phase_jac(:,6:10),'all');mean(all_phase_jac_noisy(:,6:10),'all')], ...
    'VariableNames',{'HeldOutMean_RMS_mm','BetweenCase_SD_mm','MeanFold_pct'},'RowNames',{'Clean','Perturbed'});
disp(H); writetable(H,fullfile(result_path,'R2_HeldOutPhases.csv'),'WriteRowNames',true);
save(fullfile(result_path,'Phase3_Full_Phases_Results.mat'),'all_phase_rms','all_phase_jac','all_phase_rms_noisy','all_phase_jac_noisy','alpha_kide','kappa');

function [img_size, vs] = get_dirlab_params(case_id)
    if case_id <= 5
        img_size = [256, 256, 94]; vs = [0.97, 0.97, 2.5];
    else
        img_size = [512, 512, 128]; vs = [0.97, 0.97, 2.5];
    end
    switch case_id
        case 1, img_size(3) = 94;
        case 2, img_size(3) = 112; vs(1:2) = [1.16, 1.16];
        case 3, img_size(3) = 104; vs(1:2) = [1.15, 1.15];
        case 4, img_size(3) = 99;  vs(1:2) = [1.13, 1.13];
        case 5, img_size(3) = 106; vs(1:2) = [1.10, 1.10];
        case 7, img_size(3) = 136;
        case 10, img_size(3) = 120;
    end
end
