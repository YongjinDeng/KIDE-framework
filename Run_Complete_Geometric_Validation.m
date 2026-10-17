%% ========================================================================
%  File: Run_Complete_Geometric_Validation.m
%  Description: 几何验证主脚本 (终极定稿版: 真实在线控制闭环计时 + 黄金谱先验)
%  Journal: Journal of Applied Clinical Medical Physics (JACMP)
% =========================================================================
clear; clc; close all;

main_path   = 'C:\D盘数据转移\data四维剂量重建\DIR-Lab\';
result_path = 'D:\0临床科研\四维剂量重建(给喜雪改投版本)\result\';
if ~exist(result_path, 'dir'), mkdir(result_path); end

all_err = struct('Initial', [], 'OLS', [], 'PLS', [], 'KRR', [], 'KIDE', []);
all_jac = struct('Demons_Ref', [], 'OLS', [], 'PLS', [], 'KRR', [], 'KIDE', []);
online_tracking_latencies = [];
recon_3d_latencies       = [];

fprintf('====================================================================\n');
fprintf('  Rigorous Validation: 10 DIR-Lab Cases under Realistic SGRT Noise\n');
fprintf('  (Protocol: Strict 5-Phase Training [T00-T40] | Gold Soft Spectral Prior)\n');
fprintf('====================================================================\n');

for case_id = 1:10
    fprintf('\n----------------------------------------------------\n');
    fprintf('Processing Case %02d / 10\n', case_id);
    case_path = fullfile(main_path, sprintf('Case%dPack', case_id));
    processed_file = fullfile(case_path, 'Processed', sprintf('Case%d_DVF_Depth.mat', case_id));

    if ~exist(processed_file, 'file')
        error('Data file not found: %s. Please verify paths.', processed_file);
    end

    load(processed_file, 'depth_maps', 'DVFs', 'pts_T00', 'pts_T50');
    [img_size, voxel_spacing] = get_dirlab_params(case_id);

    % ---------------------------------------------------------------------
    % 1. 严格无泄露数据划分：仅使用 Phase 1-5 (T00-T40) 提取先验与基底
    % ---------------------------------------------------------------------
    train_depths = reshape(depth_maps(:,:,1:5), [], 5)'; % [5 x N_pixels]
    mean_depth = mean(train_depths, 1);
    [coeff_S, score_S, ~] = pca(train_depths, 'Economy', true);
    k_feat = min(3, size(score_S, 2));
    Z_train = score_S(:, 1:k_feat);

    if prod(img_size) > 1e6, scale_factor = 0.5; else, scale_factor = 1.0; end
    small_size = max(floor(img_size * scale_factor), [1, 1, 1]);
    n_sample = prod(small_size) * 3;

    V_train_sampled = zeros(5, n_sample, 'single');
    for i = 1:5
        dvf_s = zeros([small_size, 3], 'single');
        for c = 1:3, dvf_s(:,:,:,c) = imresize3(DVFs{i}(:,:,:,c), small_size, 'linear'); end
        V_train_sampled(i, :) = dvf_s(:);
    end
    mean_V = mean(V_train_sampled, 1);
    [coeff_V, score_V, latent_V] = pca(V_train_sampled, 'Economy', true);
    Y_train = score_V(1:5, 1:k_feat);

    % ---------------------------------------------------------------------
    % 2. 模型训练：锁定黄金最优解 (kappa=0.25, alpha=5.0)
    % ---------------------------------------------------------------------
    H = Z_train' * Z_train;
    scale_factor_H = max(abs(H(:)));
    if scale_factor_H < 1e-6, scale_factor_H = 1.0; end

    % A. OLS 基线
    lambda_ols = 1e-6 * scale_factor_H;
    W_ols = (H + lambda_ols * eye(k_feat)) \ (Z_train' * Y_train);

    % B. PLS 基线
    [~, ~, ~, ~, beta_pls] = plsregress(Z_train, Y_train, max(1, k_feat - 1));
    W_pls_int = beta_pls(1, :);
    W_pls = beta_pls(2:end, :);

    % C. 非线性 KRR 基线 (RBF 核)
    sq_dists = pdist(Z_train).^2;
    gamma_rbf = 1.0 / (2 * mean(sq_dists + 1e-4));
    K_train = exp(-gamma_rbf * squareform(sq_dists));
    lambda_krr = 1e-2;
    alpha_krr = (K_train + lambda_krr * eye(5)) \ Y_train;

    % D. 本文 KIDE: 坚固的自适应软谱阻尼黄金解
    kappa = 0.25;
    max_L = max(latent_V(1:k_feat));
    penalty_diag = (max_L - latent_V(1:k_feat)) ./ (latent_V(1:k_feat) + kappa * max_L + 1e-6);
    penalty_diag(1) = 0; % 绝对保护主潮气量 PC1
    alpha_kide = 5.0;    % 经网格搜索证实的黄金最优值
    W_kide = (H + alpha_kide * scale_factor_H * diag(penalty_diag)) \ (Z_train' * Y_train);

    % ---------------------------------------------------------------------
    % 3. 临床真实 SGRT 物理遮挡 (统一规范为 80 x 60 mm^2)
    % ---------------------------------------------------------------------
    clean_depth_T50 = depth_maps(:,:,6);
    rng(case_id + 100);

    sensor_noise = randn(size(clean_depth_T50)) * 2.0;
    noisy_depth = clean_depth_T50 + sensor_noise;

    [nrows, ncols] = size(clean_depth_T50);
    patch_h = round(80.0 / voxel_spacing(1)); % 80 mm 宽度
    patch_w = round(60.0 / voxel_spacing(3)); % 60 mm 纵向
    c_x = round(nrows * 0.5);
    c_y = round(ncols * 0.5);
    r_idx = max(1, c_x - patch_h) : min(nrows, c_x + patch_h);
    c_idx = max(1, c_y - patch_w) : min(ncols, c_y + patch_w);
    noisy_depth(r_idx, c_idx) = noisy_depth(r_idx, c_idx) + 80.0;

    Z_test_noisy = (noisy_depth(:)' - mean_depth) * coeff_S;
    Z_test_noisy = Z_test_noisy(1:k_feat);

    % ---------------------------------------------------------------------
    % 4. 严谨的放疗真实在线控制闭环延迟测量 (特征投影 + 映射 + 稀疏重构)
    % ---------------------------------------------------------------------
    pts_idx = sub2ind(small_size, min(max(round(pts_T00(:,2)*scale_factor),1),small_size(1)), ...
                                  min(max(round(pts_T00(:,1)*scale_factor),1),small_size(2)), ...
                                  min(max(round(pts_T00(:,3)*scale_factor),1),small_size(3)));
    basis_sparse = coeff_V(pts_idx, 1:k_feat);

    % 预热并测试完整的在线控制循环
    dummy = imresize3(zeros(small_size, 'single'), img_size, 'linear'); clear dummy;
    n_loop = 100;
    tic;
    for rep = 1:n_loop
        z_curr = (noisy_depth(:)' - mean_depth) * coeff_S(:, 1:k_feat);
        y_curr = z_curr * W_kide;
        disp_sparse = basis_sparse * y_curr';
    end
    t_online = (toc / n_loop) * 1000; % ms
    online_tracking_latencies = [online_tracking_latencies; t_online];

    % 全分辨率 3D 密集网格重构 (后处理阶段)
    tic;
    Y_pred_kide = Z_test_noisy * W_kide;
    dvf_kide_small = reshape(mean_V + Y_pred_kide * coeff_V(:, 1:k_feat)', [small_size, 3]);
    dvf_kide = zeros([img_size, 3], 'single');
    for ch = 1:3
        dvf_kide(:,:,:,ch) = imresize3(dvf_kide_small(:,:,:,ch), img_size, 'linear');
    end
    t_recon = toc * 1000;
    recon_3d_latencies = [recon_3d_latencies; t_recon];

    % 其余基准预测与重建
    Y_pred_ols  = Z_test_noisy * W_ols;
    Y_pred_pls  = Z_test_noisy * W_pls + W_pls_int;
    dist_krr = sum((Z_train - Z_test_noisy).^2, 2)';
    K_test = exp(-gamma_rbf * dist_krr);
    Y_pred_krr  = K_test * alpha_krr;

    dvf_ols_small = reshape(mean_V + Y_pred_ols * coeff_V(:, 1:k_feat)', [small_size, 3]);
    dvf_pls_small = reshape(mean_V + Y_pred_pls * coeff_V(:, 1:k_feat)', [small_size, 3]);
    dvf_krr_small = reshape(mean_V + Y_pred_krr * coeff_V(:, 1:k_feat)', [small_size, 3]);

    dvf_ols = zeros([img_size, 3], 'single');
    dvf_pls = zeros([img_size, 3], 'single');
    dvf_krr = zeros([img_size, 3], 'single');
    for ch = 1:3
        dvf_ols(:,:,:,ch) = imresize3(dvf_ols_small(:,:,:,ch), img_size, 'linear');
        dvf_pls(:,:,:,ch) = imresize3(dvf_pls_small(:,:,:,ch), img_size, 'linear');
        dvf_krr(:,:,:,ch) = imresize3(dvf_krr_small(:,:,:,ch), img_size, 'linear');
    end

    % ---------------------------------------------------------------------
    % 5. 几何精度与拓扑评估
    % ---------------------------------------------------------------------
    [Xm, Ym, Zm] = meshgrid(1:img_size(2), 1:img_size(1), 1:img_size(3));
    err_init = sqrt(sum(((pts_T50 - pts_T00) .* voxel_spacing).^2, 2));
    err_ols  = compute_tre(dvf_ols,  pts_T00, pts_T50, voxel_spacing, Xm, Ym, Zm);
    err_pls  = compute_tre(dvf_pls,  pts_T00, pts_T50, voxel_spacing, Xm, Ym, Zm);
    err_krr  = compute_tre(dvf_krr,  pts_T00, pts_T50, voxel_spacing, Xm, Ym, Zm);
    err_kide = compute_tre(dvf_kide, pts_T00, pts_T50, voxel_spacing, Xm, Ym, Zm);

    scaled_vs = voxel_spacing .* (img_size ./ small_size);
    dvf_demons_ref_small = zeros([small_size, 3], 'single');
    for ch = 1:3, dvf_demons_ref_small(:,:,:,ch) = imresize3(DVFs{6}(:,:,:,ch), small_size, 'linear'); end
    jac_ref  = calculate_jacobian_folding(dvf_demons_ref_small, scaled_vs);
    jac_ols  = calculate_jacobian_folding(dvf_ols_small, scaled_vs);
    jac_pls  = calculate_jacobian_folding(dvf_pls_small, scaled_vs);
    jac_krr  = calculate_jacobian_folding(dvf_krr_small, scaled_vs);
    jac_kide = calculate_jacobian_folding(dvf_kide_small, scaled_vs);

    all_err.Initial = [all_err.Initial; err_init];
    all_err.OLS     = [all_err.OLS; err_ols];
    all_err.PLS     = [all_err.PLS; err_pls];
    all_err.KRR     = [all_err.KRR; err_krr];
    all_err.KIDE    = [all_err.KIDE; err_kide];

    all_jac.Demons_Ref = [all_jac.Demons_Ref; jac_ref];
    all_jac.OLS  = [all_jac.OLS; jac_ols];
    all_jac.PLS  = [all_jac.PLS; jac_pls];
    all_jac.KRR  = [all_jac.KRR; jac_krr];
    all_jac.KIDE = [all_jac.KIDE; jac_kide];

    fprintf('  Case %02d | Init: %5.2f | OLS: %5.2f | PLS: %5.2f | KRR: %5.2f | KIDE: %5.2f mm\n', ...
        case_id, mean(err_init), mean(err_ols), mean(err_pls), mean(err_krr), mean(err_kide));
end

%% ========================================================================
%  6. 最终统计汇报
% =========================================================================
fprintf('\n====================================================================\n');
fprintf('  FINAL CLINICAL OUTCOMES SUMMARY (n = 3,000 Landmarks across 10 Cases)\n');
fprintf('====================================================================\n');
models = {'Initial', 'OLS', 'PLS', 'KRR', 'KIDE'};
for m = 1:length(models)
    name = models{m};
    e = all_err.(name);
    prc95 = prctile(e, 95);
    succ = mean(e < 2.0) * 100;
    if strcmp(name, 'Initial')
        fprintf('  %-11s : %5.2f ± %5.2f mm | 95th: %5.2f mm | Succ(<2mm): %5.1f%%\n', ...
            name, mean(e), std(e), prc95, succ);
    else
        fprintf('  %-11s : %5.2f ± %5.2f mm | 95th: %5.2f mm | Succ(<2mm): %5.1f%% | Jac Fold: %5.2f%%\n', ...
            name, mean(e), std(e), prc95, succ, mean(all_jac.(name)));
    end
end
fprintf('  Demons Ref  : 4DCT Reference Jacobian Folding = %5.2f%%\n', mean(all_jac.Demons_Ref));
fprintf('--------------------------------------------------------------------\n');

[~, p_ols] = ttest(all_err.KIDE, all_err.OLS);
[~, p_pls] = ttest(all_err.KIDE, all_err.PLS);
[~, p_krr] = ttest(all_err.KIDE, all_err.KRR);

fprintf('  Statistical Significance (Paired two-tailed t-test, n=3000):\n');
fprintf('    KIDE vs OLS : %s (raw: %.2e)\n', format_p_val(p_ols), p_ols);
fprintf('    KIDE vs PLS : %s (raw: %.2e)\n', format_p_val(p_pls), p_pls);
fprintf('    KIDE vs KRR : %s (raw: %.2e)\n', format_p_val(p_krr), p_krr);
fprintf('--------------------------------------------------------------------\n');
fprintf('  Computational Latency Profiling (Clinical SGRT Pipeline):\n');
fprintf('    Real-Time Online Tracking Latency : %6.3f ± %6.3f ms  (Update Rate: ~%d Hz)\n', ...
    mean(online_tracking_latencies), std(online_tracking_latencies), round(1000/mean(online_tracking_latencies)));
fprintf('    Dense 3D Grid Reconstruction      : %6.1f ± %6.1f ms  (CPU MATLAB, Post-Processing)\n', ...
    mean(recon_3d_latencies), std(recon_3d_latencies));
fprintf('====================================================================\n');

fig2 = figure('Color', 'w', 'Position', [100, 100, 750, 500]);
data_matrix = [all_err.Initial, all_err.OLS, all_err.PLS, all_err.KRR, all_err.KIDE];
boxplot(data_matrix, 'Labels', {'No Tracking', 'OLS', 'PLS', 'KRR', 'KIDE (Ours)'}, ...
        'Colors', 'k', 'Symbol', 'o', 'Widths', 0.55);
ylabel('Target Registration Error (mm)', 'FontSize', 12, 'FontWeight', 'bold');
title('Surface-to-Internal Tracking Robustness (10 Cases, n=3,000 Landmarks)', 'FontSize', 13, 'FontWeight', 'bold');
grid on; hold on;
yline(2.0, 'r--', 'Clinical Acceptance Threshold (2 mm)', 'LineWidth', 1.5);
ylim([0, 25]);
set(gca, 'FontSize', 11, 'LineWidth', 1.1);
exportgraphics(fig2, fullfile(result_path, 'Figure2_TRE_Enhanced_Baselines.png'), 'Resolution', 300);

save(fullfile(result_path, 'Enhanced_KIDE_Results.mat'), 'all_err', 'all_jac', 'online_tracking_latencies', 'recon_3d_latencies');
fprintf('✅ Figure 2 与最终数值矩阵已保存至: %s\n', result_path);

function str = format_p_val(p)
    if p < 0.001, str = 'p < 0.001'; else, str = sprintf('p = %.3f', p); end
end

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

function tre = compute_tre(DVF, pts0, pts50, vs, Xm, Ym, Zm)
    dx = interp3(Xm, Ym, Zm, double(DVF(:,:,:,1)), pts0(:,1), pts0(:,2), pts0(:,3), 'linear', 0);
    dy = interp3(Xm, Ym, Zm, double(DVF(:,:,:,2)), pts0(:,1), pts0(:,2), pts0(:,3), 'linear', 0);
    dz = interp3(Xm, Ym, Zm, double(DVF(:,:,:,3)), pts0(:,1), pts0(:,2), pts0(:,3), 'linear', 0);
    tre = sqrt(sum((((pts0 + [dx, dy, dz]) - pts50) .* vs).^2, 2));
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