%% ========================================================================
%  Phase 1: SGRT 结构性遮挡噪声模拟与多基线全面评估 (微调统一噪声版)
%  全面指标: Mean TRE, 95th Percentile TRE, Success Rate (<2mm), 3D Jacobian
% =========================================================================
clear; clc; close all;

main_path = 'D:/0临床科研/四维剂量重建/data/';

% 存储全局 3000 个点的误差
all_err = struct('Initial', [], 'OLS', [], 'PLS', [], 'KIDE', []);
all_jac = struct('OLS', [], 'PLS', [], 'KIDE', []);

fprintf('==================================================\n');
fprintf('Phase 1: SGRT 潜空间高频噪声模拟与多基线全面评估\n');
fprintf('==================================================\n');

for case_id = 1:10
    case_path = fullfile(main_path, sprintf('Case%dPack/', case_id));
    processed_file = fullfile(case_path, 'Processed', sprintf('Case%d_DVF_Depth.mat', case_id));
    if ~exist(processed_file, 'file')
        fprintf('  ⚠️ Case %d: 数据未预处理，跳过。\n', case_id);
        continue; 
    end
    
    load(processed_file, 'depth_maps', 'DVFs', 'pts_T00', 'pts_T50');
    [img_size, voxel_spacing] = get_dirlab_params(case_id);
    
    % --- 1. 特征提取 (仅 Phase 1-5) ---
    train_depths = reshape(depth_maps(:,:,1:5), [], 5)';
    [coeff_S, score_S, ~] = pca(train_depths, 'Economy', true);
    k_feat = min(3, size(score_S, 2)); 
    Z_train = score_S(:, 1:k_feat); 
    mean_depth = mean(train_depths, 1);
    
    % --- 2. 3D空间降采样 (仅 Phase 1-5) ---
    if prod(img_size) > 1e6, scale_factor = 0.5; else, scale_factor = 1.0; end
    small_size = max(floor(img_size * scale_factor), [1, 1, 1]);
    
    V_train_sampled = zeros(5, prod(small_size)*3, 'single');
    for i = 1:5
        dvf_small = zeros([small_size, 3], 'single');
        for c = 1:3
            dvf_small(:,:,:,c) = imresize3(DVFs{i}(:,:,:,c), small_size, 'linear');
        end
        V_train_sampled(i, :) = dvf_small(:);
    end
    mean_V = mean(V_train_sampled, 1);
    [coeff_V, score_V, latent_V] = pca(V_train_sampled, 'Economy', true);
    Y_train = score_V(1:5, 1:k_feat); 
    
    % --- 3. 多模型训练 (OLS, PLS, KIDE) ---
    W_ols = (Z_train'*Z_train + 1e-6*eye(k_feat)) \ (Z_train'*Y_train);
    
    [~, ~, ~, ~, beta_pls] = plsregress(Z_train, Y_train, max(1, k_feat - 1));
    W_pls_int = beta_pls(1,:); 
    W_pls = beta_pls(2:end,:);
    
    H = Z_train'*Z_train; 
    scale_H = max(abs(H(:))); 
    if scale_H < 1e-6, scale_H = 1; end
    penalty_diag = 1 ./ (latent_V(1:k_feat) / max(latent_V(1:k_feat)) + 1e-4);
    penalty_diag(1) = 0;
    W_kide = (H + 5.0 * scale_H * diag(penalty_diag)) \ (Z_train'*Y_train);
    
    % --- 4. 注入潜空间高频噪声 (统一噪声) ---
    Z_test_clean = (reshape(depth_maps(:,:,6), 1, []) - mean_depth) * coeff_S;
    Z_test_clean = Z_test_clean(1:k_feat);
    
    rng(case_id); 
    noise = randn(1, k_feat) .* std(Z_train) .* [0.1, 2.0, 5.0]; 
    Z_test_noisy = Z_test_clean + noise;
    
    % --- 5. 推断与 3D 上采样 ---
    Y_pred_ols = Z_test_noisy * W_ols; 
    Y_pred_pls = Z_test_noisy * W_pls + W_pls_int;
    Y_pred_kide = Z_test_noisy * W_kide;
    
    dvf_ols_small = reshape(mean_V + Y_pred_ols * coeff_V(:, 1:k_feat)', [small_size, 3]);
    dvf_pls_small = reshape(mean_V + Y_pred_pls * coeff_V(:, 1:k_feat)', [small_size, 3]);
    dvf_kide_small = reshape(mean_V + Y_pred_kide * coeff_V(:, 1:k_feat)', [small_size, 3]);
    
    dvf_ols = zeros([img_size, 3], 'single');
    dvf_pls = zeros([img_size, 3], 'single');
    dvf_kide = zeros([img_size, 3], 'single');
    for ch = 1:3
        dvf_ols(:,:,:,ch) = imresize3(dvf_ols_small(:,:,:,ch), img_size, 'linear');
        dvf_pls(:,:,:,ch) = imresize3(dvf_pls_small(:,:,:,ch), img_size, 'linear');
        dvf_kide(:,:,:,ch) = imresize3(dvf_kide_small(:,:,:,ch), img_size, 'linear');
    end
    
    % --- 6. 全面几何与拓扑验证 ---
    [Xm, Ym, Zm] = meshgrid(1:img_size(2), 1:img_size(1), 1:img_size(3));
    err_init = sqrt(sum(((pts_T50 - pts_T00) .* voxel_spacing).^2, 2));
    err_ols  = compute_tre(dvf_ols, pts_T00, pts_T50, voxel_spacing, Xm, Ym, Zm);
    err_pls  = compute_tre(dvf_pls, pts_T00, pts_T50, voxel_spacing, Xm, Ym, Zm);
    err_kide = compute_tre(dvf_kide, pts_T00, pts_T50, voxel_spacing, Xm, Ym, Zm);
    
    scaled_vs = voxel_spacing .* (img_size ./ small_size);
    jac_ols = calculate_jacobian_folding(dvf_ols_small, scaled_vs);
    jac_pls = calculate_jacobian_folding(dvf_pls_small, scaled_vs);
    jac_kide = calculate_jacobian_folding(dvf_kide_small, scaled_vs);
    
    all_err.Initial = [all_err.Initial; err_init]; 
    all_err.OLS = [all_err.OLS; err_ols]; all_err.PLS = [all_err.PLS; err_pls]; all_err.KIDE = [all_err.KIDE; err_kide];
    all_jac.OLS = [all_jac.OLS; jac_ols]; all_jac.PLS = [all_jac.PLS; jac_pls]; all_jac.KIDE = [all_jac.KIDE; jac_kide];
    
    fprintf('  Case %02d | TRE: OLS = %.2f | PLS = %.2f | KIDE = %.2f mm\n', ...
        case_id, mean(err_ols), mean(err_pls), mean(err_kide));
end

% --- 生成 Boxplot ---
fig1 = figure('Color', 'w', 'Position', [100, 100, 650, 500]);
group_data = [all_err.Initial; all_err.OLS; all_err.PLS; all_err.KIDE];
group_labels = [ones(length(all_err.Initial),1); 2*ones(length(all_err.OLS),1); ...
                3*ones(length(all_err.PLS),1); 4*ones(length(all_err.KIDE),1)];
boxplot(group_data, group_labels, 'Labels', {'No Tracking', 'OLS', 'PLS', 'KIDE'}, ...
        'Colors', [0.2 0.2 0.2; 0.8 0.2 0.2; 0.2 0.5 0.8; 0.2 0.6 0.2], 'Symbol', 'o');
ylabel('TRE (mm)', 'FontSize', 12, 'FontWeight', 'bold');
title('Tracking Robustness under Occlusion Noise (n=3,000)', 'FontSize', 14);
yline(2.0, 'r--', 'Clinical Threshold', 'LineWidth', 1.5);
grid on; set(gca, 'FontSize', 11); ylim([0, 30]);
exportgraphics(fig1, fullfile(main_path, 'Figure2_TRE_Boxplot.png'), 'Resolution', 300);
fprintf('  ✅ Figure 2 已保存\n');

%% --- 全局统计输出 ---
fprintf('\n==================================================\n');
fprintf('临床结果大汇总 (n=3000 Landmarks)\n');
fprintf('  No Tracking : %.2f ± %.2f mm | Success(<2mm): %5.1f%%\n', mean(all_err.Initial), std(all_err.Initial), mean(all_err.Initial < 2.0)*100);
fprintf('  OLS Error   : %.2f ± %.2f mm | Success(<2mm): %5.1f%% | Jac Fold: %.2f%%\n', mean(all_err.OLS), std(all_err.OLS), mean(all_err.OLS < 2.0)*100, mean(all_jac.OLS));
fprintf('  PLS Error   : %.2f ± %.2f mm | Success(<2mm): %5.1f%% | Jac Fold: %.2f%%\n', mean(all_err.PLS), std(all_err.PLS), mean(all_err.PLS < 2.0)*100, mean(all_jac.PLS));
fprintf('  KIDE Error  : %.2f ± %.2f mm | Success(<2mm): %5.1f%% | Jac Fold: %.2f%%\n', mean(all_err.KIDE), std(all_err.KIDE), mean(all_err.KIDE < 2.0)*100, mean(all_jac.KIDE));
fprintf('==================================================\n');

[~, p_ols] = ttest(all_err.KIDE, all_err.OLS);
[~, p_pls] = ttest(all_err.KIDE, all_err.PLS);
fprintf('  KIDE vs OLS: p = %.2e\n', p_ols);
fprintf('  KIDE vs PLS: p = %.2e\n', p_pls);
fprintf('==================================================\n');

%% --- 辅助函数 ---
function jac_fold = calculate_jacobian_folding(DVF, vs)
    [du1_dcol, du1_drow, du1_dslice] = gradient(DVF(:,:,:,1), vs(1), vs(2), vs(3));
    [du2_dcol, du2_drow, du2_dslice] = gradient(DVF(:,:,:,2), vs(1), vs(2), vs(3));
    [du3_dcol, du3_drow, du3_dslice] = gradient(DVF(:,:,:,3), vs(1), vs(2), vs(3));
    J11 = 1 + du1_dcol; J12 = du1_drow; J13 = du1_dslice;
    J21 = du2_dcol; J22 = 1 + du2_drow; J23 = du2_dslice;
    J31 = du3_dcol; J32 = du3_drow; J33 = 1 + du3_dslice;
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
    if case_id<=5, img_size=[256, 256, 94]; vs=[0.97, 0.97, 2.5]; else, img_size=[512, 512, 128]; vs=[0.97, 0.97, 2.5]; end
    switch case_id
        case 1, img_size(3)=94; case 2, img_size(3)=112; vs(1:2)=[1.16, 1.16];
        case 3, img_size(3)=104; vs(1:2)=[1.15, 1.15]; case 4, img_size(3)=99; vs(1:2)=[1.13, 1.13];
        case 5, img_size(3)=106; vs(1:2)=[1.10, 1.10]; case 7, img_size(3)=136; case 10, img_size(3)=120;
    end
end
