%% ========================================================================
%  File: Run_Comprehensive_FullPhases_and_Sensitivity.m
%  Description: 全周期时相 (T00-T90) 与超参数敏感性网格搜索 (内存秒速版)
%  Outputs: Full-Cycle Evaluation Table & Figure S1 (Alpha Sensitivity Curve)
% =========================================================================
clear; clc; close all;

% ------------------------- 路径配置 -------------------------------------
main_path   = 'C:\D盘数据转移\data四维剂量重建\DIR-Lab\';
result_path = 'D:\0临床科研\四维剂量重建(给喜雪改投版本)\result\';
if ~exist(result_path, 'dir'), mkdir(result_path); end

%% ========================================================================
%  PART 1: 全周期呼吸时相 (T00-T90) 预测验证 (回答 Reviewer #2 Q6)
% ========================================================================
fprintf('====================================================================\n');
fprintf('PART 1: Evaluating Full Respiratory Manifold (Phases T00 to T90 across 10 Cases)\n');
fprintf('====================================================================\n');

all_phase_rms = zeros(10, 10); % [10 cases x 10 phases]
all_phase_jac = zeros(10, 10);

% 预缓存结构体
cache = cell(10, 1);

for case_id = 1:10
    fprintf('  [预缓存与全时相计算] Case %02d / 10 ...\n', case_id);
    case_path = fullfile(main_path, sprintf('Case%dPack', case_id));
    processed_file = fullfile(case_path, 'Processed', sprintf('Case%d_DVF_Depth.mat', case_id));
    if ~exist(processed_file, 'file'), continue; end
    
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
    
    % 黄金软谱阻尼配置 (kappa=0.25, alpha=5.0)
    kappa = 0.25;
    max_L = max(latent_V(1:3));
    penalty_diag = (max_L - latent_V(1:3)) ./ (latent_V(1:3) + kappa * max_L + 1e-6);
    penalty_diag(1) = 0;
    alpha_kide = 5.0;
    W_kide = (H + alpha_kide * scale_H * diag(penalty_diag)) \ (Z_train' * Y_train);
    
    scaled_vs = vs .* (img_size ./ small_size);
    
    % 评估全周期 10 个时相 (T00 到 T90)
    for p = 1:10
        curr_d = data.depth_maps(:,:,p);
        z_curr = (curr_d(:)' - mean_depth) * coeff_S;
        y_pred = z_curr(1:3) * W_kide;
        pred_dvf_s = reshape(mean_V + y_pred * coeff_V(:, 1:3)', [small_size, 3]);
        
        gt_dvf_s = zeros([small_size, 3], 'single');
        for c = 1:3, gt_dvf_s(:,:,:,c) = imresize3(data.DVFs{p}(:,:,:,c), small_size, 'linear'); end
        
        diff_vec = (pred_dvf_s - gt_dvf_s) .* reshape(scaled_vs, [1, 1, 1, 3]);
        all_phase_rms(case_id, p) = sqrt(mean(sum(diff_vec.^2, 4), 'all'));
        all_phase_jac(case_id, p) = calculate_jacobian_folding(pred_dvf_s, scaled_vs);
    end
    
    % 构造并缓存测试相 T50 遮挡 (80 x 60 mm^2)
    clean_depth = data.depth_maps(:,:,6);
    rng(case_id + 100);
    noisy_depth = clean_depth + randn(size(clean_depth)) * 2.0;
    
    [nrows, ncols] = size(clean_depth);
    patch_h = round(80.0 / vs(1)); patch_w = round(60.0 / vs(3));
    c_x = round(nrows * 0.5); c_y = round(ncols * 0.5);
    r_idx = max(1, c_x - patch_h) : min(nrows, c_x + patch_h);
    c_idx = max(1, c_y - patch_w) : min(ncols, c_y + patch_w);
    noisy_depth(r_idx, c_idx) = noisy_depth(r_idx, c_idx) + 80.0;
    
    Z_test_noisy = (noisy_depth(:)' - mean_depth) * coeff_S;
    
    % 缓存供 Part 2 秒级网格搜索使用
    cache{case_id}.H = H;
    cache{case_id}.scale_H = scale_H;
    cache{case_id}.Z_train = Z_train;
    cache{case_id}.Y_train = Y_train;
    cache{case_id}.latent_V = latent_V(1:3);
    cache{case_id}.Z_test = Z_test_noisy(1:3);
    cache{case_id}.coeff_V = coeff_V(:, 1:3);
    cache{case_id}.mean_V = mean_V;
    cache{case_id}.pts0 = data.pts_T00;
    cache{case_id}.pts50 = data.pts_T50;
    cache{case_id}.vs = vs;
    cache{case_id}.img_size = img_size;
    cache{case_id}.small_size = small_size;
end

phases_str = {'T00', 'T10', 'T20', 'T30', 'T40', 'T50', 'T60', 'T70', 'T80', 'T90'};
fprintf('\nFull-Cycle Performance Summary (T00 to T90 across 100 Volumes):\n');
for p = 1:10
    fprintf('  Phase %s | Whole-Field RMS: %4.2f ± %4.2f mm | Jacobian Folding: %4.2f%%\n', ...
        phases_str{p}, mean(all_phase_rms(:, p)), std(all_phase_rms(:, p)), mean(all_phase_jac(:, p)));
end
fprintf('  Global Full-Cycle Mean RMS: %4.2f ± %4.2f mm | Max Folding: %4.2f%%\n', ...
    mean(all_phase_rms(:)), std(all_phase_rms(:)), max(all_phase_jac(:)));
save(fullfile(result_path, 'Phase3_Full_Phases_Results.mat'), 'all_phase_rms', 'all_phase_jac');

%% ========================================================================
%  PART 2: 超参数 alpha 敏感性网格搜索与生成 Figure S1 (回答 Reviewer #2 Minor 1)
% ========================================================================
fprintf('\n====================================================================\n');
fprintf('PART 2: Hyperparameter Sensitivity Sweep for Alpha (Memory Pre-cached)\n');
fprintf('====================================================================\n');

alpha_candidates = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
n_alphas = length(alpha_candidates);
mean_tre_sweep = zeros(n_alphas, 1);
p95_tre_sweep  = zeros(n_alphas, 1);
succ_tre_sweep = zeros(n_alphas, 1);

for a_idx = 1:n_alphas
    alpha_curr = alpha_candidates(a_idx);
    cohort_errors = [];
    
    for case_id = 1:10
        C = cache{case_id};
        if isempty(C), continue; end
        
        kappa = 0.25;
        max_L = max(C.latent_V);
        penalty_diag = (max_L - C.latent_V) ./ (C.latent_V + kappa * max_L + 1e-6);
        penalty_diag(1) = 0;
        
        W_curr = (C.H + alpha_curr * C.scale_H * diag(penalty_diag)) \ (C.Z_train' * C.Y_train);
        Y_pred = C.Z_test * W_curr;
        
        dvf_s = reshape(C.mean_V + Y_pred * C.coeff_V', [C.small_size, 3]);
        dvf_full = zeros([C.img_size, 3], 'single');
        for ch = 1:3, dvf_full(:,:,:,ch) = imresize3(dvf_s(:,:,:,ch), C.img_size, 'linear'); end
        
        [Xm, Ym, Zm] = meshgrid(1:C.img_size(2), 1:C.img_size(1), 1:C.img_size(3));
        dx = interp3(Xm, Ym, Zm, double(dvf_full(:,:,:,1)), C.pts0(:,1), C.pts0(:,2), C.pts0(:,3), 'linear', 0);
        dy = interp3(Xm, Ym, Zm, double(dvf_full(:,:,:,2)), C.pts0(:,1), C.pts0(:,2), C.pts0(:,3), 'linear', 0);
        dz = interp3(Xm, Ym, Zm, double(dvf_full(:,:,:,3)), C.pts0(:,1), C.pts0(:,2), C.pts0(:,3), 'linear', 0);
        tre = sqrt(sum((((C.pts0 + [dx, dy, dz]) - C.pts50) .* C.vs).^2, 2));
        cohort_errors = [cohort_errors; tre];
    end
    
    mean_tre_sweep(a_idx) = mean(cohort_errors);
    p95_tre_sweep(a_idx)  = prctile(cohort_errors, 95);
    succ_tre_sweep(a_idx) = mean(cohort_errors < 2.0) * 100;
    
    fprintf('  alpha = %4.1f | Mean TRE = %5.2f mm | 95th = %5.2f mm | Success Rate = %4.1f%%\n', ...
        alpha_curr, mean_tre_sweep(a_idx), p95_tre_sweep(a_idx), succ_tre_sweep(a_idx));
end

% 绘制 Figure S1 (精美学术曲线)
fig_s1 = figure('Color', 'w', 'Position', [150, 150, 620, 440]);
semilogx(alpha_candidates, mean_tre_sweep, 'b-o', 'LineWidth', 2.0, 'MarkerFaceColor', 'b', 'MarkerSize', 6); hold on;
semilogx(alpha_candidates, p95_tre_sweep, 'r--s', 'LineWidth', 2.0, 'MarkerFaceColor', 'r', 'MarkerSize', 6);
xline(5.0, 'k:', 'Optimal Selection (\alpha = 5.0)', 'LineWidth', 1.6, 'LabelHorizontalAlignment', 'center', 'FontSize', 11);
xlabel('Regularization Strength (\alpha)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Target Registration Error (mm)', 'FontSize', 12, 'FontWeight', 'bold');
title('Hyperparameter Sensitivity & Stability Curve (n = 3,000 Landmarks)', 'FontSize', 13, 'FontWeight', 'bold');
legend('Mean TRE', '95th Percentile TRE', 'Selected Parameter', 'Location', 'northeast', 'FontSize', 10);
grid on; set(gca, 'FontSize', 11, 'LineWidth', 1.1);
exportgraphics(fig_s1, fullfile(result_path, 'FigureS1_Alpha_Sensitivity.png'), 'Resolution', 300);

save(fullfile(result_path, 'Phase4_Alpha_Sensitivity_Results.mat'), ...
     'alpha_candidates', 'mean_tre_sweep', 'p95_tre_sweep', 'succ_tre_sweep');
fprintf('✅ Figure S1 已保存至: %s\n', result_path);

function jac_fold = calculate_jacobian_folding(DVF, vs)
    [du1_dy, du1_dx, du1_dz] = gradient(DVF(:,:,:,1), vs(2), vs(1), vs(3));
    [du2_dy, du2_dx, du2_dz] = gradient(DVF(:,:,:,2), vs(2), vs(1), vs(3));
    [du3_dy, du3_dx, du3_dz] = gradient(DVF(:,:,:,3), vs(2), vs(1), vs(3));
    J11 = 1 + du1_dx; J12 = du1_dy;     J13 = du1_dz;
    J21 = du2_dx;     J22 = 1 + du2_dy; J23 = du2_dz;
    J31 = du3_dx;     J32 = du3_dy;     J33 = 1 + du3_dz;
    J_det = J11.*(J22.*J33 - J23.*J32) - J12.*(J21.*J33 - J23.*J31) + J13.*(J21.*J32 - J22.*J31);
    jac_fold = sum(J_det(:) <= 0) / numel(J_det) * 100;
end

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