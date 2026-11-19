%% ========================================================================
%  File: Run_Complete_Dosimetric_Validation.m
%  Description: 剂量学验证终极优化引擎 (严格对齐图像网格、彻底杜绝等高线错位)
%  Journal: Journal of Applied Clinical Medical Physics (JACMP)
% =========================================================================
clear; clc; close all;

main_path   = 'E:\D盘数据转移\data四维剂量重建\DIR-Lab\';
result_path = 'D:\0临床科研\四维剂量重建(给喜雪改投版本)\result\R2\';
if ~exist(result_path, 'dir'), mkdir(result_path); end
[alpha_kide,kappa] = R2_LoadParameters(result_path);

target_cases = [1, 5, 8];
dose_rows = [];

fprintf('====================================================================\n');
fprintf('Heterogeneity-Aware Dosimetric Surrogate (R2)\n');
fprintf('====================================================================\n');

for case_id = target_cases
    fprintf('\n----------------------------------------------------\n');
    fprintf('Processing Dosimetric Verification for Case %02d...\n', case_id);
    case_path = fullfile(main_path, sprintf('Case%dPack', case_id));
    [img_size, voxel_spacing] = get_dirlab_params(case_id);

    processed_file = fullfile(case_path, 'Processed', sprintf('Case%d_DVF_Depth.mat', case_id));
    assert(isfile(processed_file),'Missing cache %s',processed_file);
    load(processed_file, 'depth_maps', 'DVFs', 'pts_T00', 'pts_T50');
    dvf_gt_raw = DVFs{6}; % T50 Demons reference DVF

    % ---------------------------------------------------------------------
    % 读取参考相 CT 并统一转为标准矩阵坐标系 [N_rows, N_cols, N_slices]
    % ---------------------------------------------------------------------
    img_files = dir(fullfile(case_path, 'Images', '*T00*.img'));
    assert(~isempty(img_files),'Missing T00 CT');
    fid = fopen(fullfile(img_files(1).folder, img_files(1).name), 'r');
    img_raw = reshape(fread(fid, prod(img_size), 'int16'), img_size);
    fclose(fid);
    
    % 与 DIR-Lab 预处理完全保持一致的标准网格化
    img_T00 = permute(img_raw, [2, 1, 3]); 
    [H_img, W_img, D_img] = size(img_T00);

    % ---------------------------------------------------------------------
    % 1. PTV 锚定在真实具有呼吸位移的下叶活跃区
    % ---------------------------------------------------------------------
    motion_3d = sqrt(sum(((pts_T50 - pts_T00) .* voxel_spacing).^2, 2));
    active_idx = find(motion_3d >= prctile(motion_3d, 70));
    [~, sub_idx] = max(pts_T00(active_idx, 3)); 
    rep_idx = active_idx(sub_idx);

    % pts 是 [x(col), y(row), z(slice)]
    ptv_center_x = pts_T00(rep_idx, 1);
    ptv_center_y = pts_T00(rep_idx, 2);
    ptv_center_z = pts_T00(rep_idx, 3);

    % 构建标准 3D 网格: X 对应列(1:W), Y 对应行(1:H), Z 对应切片(1:D)
    [X, Y, Z] = meshgrid(1:W_img, 1:H_img, 1:D_img);

    ptv_radius = 15.0; % mm
    dist_map = sqrt(((X - ptv_center_x) * voxel_spacing(1)).^2 + ...
                    ((Y - ptv_center_y) * voxel_spacing(2)).^2 + ...
                    ((Z - ptv_center_z) * voxel_spacing(3)).^2);
    ptv_mask = (dist_map <= ptv_radius);

    n_ptv_vox = sum(ptv_mask(:));
    fprintf('  PTV Target: Center=[%.1f, %.1f, %.1f] | Volume=%d voxels | Motion=%.1f mm\n', ...
        ptv_center_x, ptv_center_y, ptv_center_z, n_ptv_vox, motion_3d(rep_idx));

    % 组织密度异质性调制
    density_map = max(0.1, min(1.5, (double(img_T00) + 1000) / 1000));
    eff_dist = dist_map .* (1.0 + 0.15 * (1.0 - density_map));

    presc_dose = 62.0; margin = 3.0; steepness = 1.5;
    plan_dose = presc_dose ./ (1 + exp(steepness * (eff_dist - (ptv_radius + margin))));
    plan_dose(plan_dose > presc_dose) = presc_dose;

    lung_mask = (img_T00 > -950) & (img_T00 < -350);

    % ---------------------------------------------------------------------
    % 2. 严格无泄露建模 (仅使用 Phase 1-5)
    % ---------------------------------------------------------------------
    train_depths = reshape(depth_maps(:,:,1:5), [], 5)';
    mean_depth = mean(train_depths, 1);
    [coeff_S, score_S, ~] = pca(train_depths, 'Economy', true);
    Z_train = score_S(:, 1:3);

    if prod(img_size) > 1e6, scale_factor = 0.5; else, scale_factor = 1.0; end
    small_size = max(floor([H_img, W_img, D_img] * scale_factor), [1, 1, 1]);

    V_train_sampled = zeros(5, prod(small_size)*3, 'single');
    for i = 1:5
        dvf_s = zeros([small_size, 3], 'single');
        for c = 1:3, dvf_s(:,:,:,c) = imresize3(DVFs{i}(:,:,:,c), small_size, 'linear'); end
        V_train_sampled(i, :) = dvf_s(:);
    end
    mean_V = mean(V_train_sampled, 1);
    [coeff_V, score_V, latent_V] = pca(V_train_sampled, 'Economy', true);
    Y_train = score_V(1:5, 1:3);

    H_mat = Z_train' * Z_train;
    scale_H = max(abs(H_mat(:)));
    if scale_H < 1e-6, scale_H = 1.0; end

    W_ols = (H_mat + 1e-6 * scale_H * eye(3)) \ (Z_train' * Y_train);

    % 训练集交叉验证软谱参数
    % R2: kappa selected using training-only CV.
    max_L = max(latent_V(1:3));
    penalty_diag = (max_L - latent_V(1:3)) ./ (latent_V(1:3) + kappa * max_L + 1e-6);
    penalty_diag(1) = 0;
    % R2: alpha selected using training-only CV.
    W_kide = (H_mat + alpha_kide * scale_H * diag(penalty_diag)) \ (Z_train' * Y_train);

    % ---------------------------------------------------------------------
    % 3. 注入临床真实视场遮挡 (80 x 60 mm^2)
    % ---------------------------------------------------------------------
    clean_depth = depth_maps(:,:,6);
    noisy_depth = R2_PerturbDepth(clean_depth,voxel_spacing,case_id+100);

    Z_test_noisy = (noisy_depth(:)' - mean_depth) * coeff_S;
    Z_test_noisy = Z_test_noisy(1:3);

    Y_ols  = Z_test_noisy * W_ols;
    Y_kide = Z_test_noisy * W_kide;

    dvf_ols_s  = reshape(mean_V + Y_ols  * coeff_V(:, 1:3)', [small_size, 3]);
    dvf_kide_s = reshape(mean_V + Y_kide * coeff_V(:, 1:3)', [small_size, 3]);

    dvf_ols  = zeros([H_img, W_img, D_img, 3], 'single');
    dvf_kide = zeros([H_img, W_img, D_img, 3], 'single');
    for ch = 1:3
        dvf_ols(:,:,:,ch)  = imresize3(dvf_ols_s(:,:,:,ch),  [H_img, W_img, D_img], 'linear');
        dvf_kide(:,:,:,ch) = imresize3(dvf_kide_s(:,:,:,ch), [H_img, W_img, D_img], 'linear');
    end

    % ---------------------------------------------------------------------
    % 4. 严格拉回形变剂量计算 (Pull-Back Sampling)
    % ---------------------------------------------------------------------
    % DVF: 通道 1 沿 X (列), 通道 2 沿 Y (行), 通道 3 沿 Z (切片)
    warp_dose = @(DVF) interp3(X, Y, Z, plan_dose, ...
        X + double(DVF(:,:,:,1)), ...
        Y + double(DVF(:,:,:,2)), ...
        Z + double(DVF(:,:,:,3)), 'linear', 0);

    dose_gt   = warp_dose(dvf_gt_raw);
    dose_ols  = warp_dose(dvf_ols);
    dose_kide = warp_dose(dvf_kide);

    calc_D95 = @(d) prctile(d(ptv_mask), 5.0);
    calc_D98 = @(d) prctile(d(ptv_mask), 2.0);
    calc_V20 = @(d) sum(d(lung_mask) >= 20.0) / (sum(lung_mask(:)) + 1e-6) * 100;

    gamma_33_ols  = compute_gamma_3d(dose_ols,  dose_gt, ptv_mask, 3.0, 1.86, voxel_spacing);
    gamma_33_kide = compute_gamma_3d(dose_kide, dose_gt, ptv_mask, 3.0, 1.86, voxel_spacing);
    gamma_22_ols  = compute_gamma_3d(dose_ols,  dose_gt, ptv_mask, 2.0, 1.24, voxel_spacing);
    gamma_22_kide = compute_gamma_3d(dose_kide, dose_gt, ptv_mask, 2.0, 1.24, voxel_spacing);

    fprintf('  D95%% (覆盖指标) : Plan=%.1f | Demons Ref=%.1f | OLS=%.1f | KIDE=%.1f Gy\n', ...
        calc_D95(plan_dose), calc_D95(dose_gt), calc_D95(dose_ols), calc_D95(dose_kide));
    fprintf('  D98%% (极值覆盖) : Plan=%.1f | Demons Ref=%.1f | OLS=%.1f | KIDE=%.1f Gy\n', ...
        calc_D98(plan_dose), calc_D98(dose_gt), calc_D98(dose_ols), calc_D98(dose_kide));
    fprintf('  V20  (肺部毒性) : Plan=%.1f%% | Demons Ref=%.1f%% | OLS=%.1f%% | KIDE=%.1f%%\n', ...
        calc_V20(plan_dose), calc_V20(dose_gt), calc_V20(dose_ols), calc_V20(dose_kide));
    fprintf('  Gamma 3mm/3%%    : OLS = %5.1f%% | KIDE = %5.1f%%\n', gamma_33_ols, gamma_33_kide);
    fprintf('  Gamma 2mm/2%%    : OLS = %5.1f%% | KIDE = %5.1f%%\n', gamma_22_ols, gamma_22_kide);

    % Preserve per-case numerical endpoints for the manuscript.
    maps={plan_dose,dose_gt,dose_ols,dose_kide};
    for m=1:4
        dose_rows=[dose_rows; case_id,m,motion_3d(rep_idx),calc_D95(maps{m}),calc_D98(maps{m}),calc_V20(maps{m}),NaN,NaN]; %#ok<AGROW>
    end
    dose_rows(end-3:end,7:8)=[NaN NaN;NaN NaN;gamma_33_ols gamma_22_ols;gamma_33_kide gamma_22_kide];
    % ---------------------------------------------------------------------
    % 5. Figure 3: Dose Map (标准横断位: 胸前壁在上，脊柱后壁在下)
    % ---------------------------------------------------------------------
    slice_z = round(ptv_center_z);
    fig3 = figure('Color', 'w', 'Position', [50, 100, 1400, 350]);

    % 标准放射学方向校正：
    % 保持原始解剖朝向 (去掉错误的 flipud，恢复胸骨朝上、脊柱朝下)
    orient_func = @(mat2d) double(mat2d); 

    img_c  = orient_func(img_T00(:,:,slice_z));
    ptv_c  = orient_func(ptv_mask(:,:,slice_z));
    w_min  = prctile(img_c(:), 2); 
    w_max  = prctile(img_c(:), 98);

    % Subplot 1: 解剖图与 PTV
    subplot(1,4,1); 
    imagesc(1:W_img, 1:H_img, img_c); colormap(gca, 'gray'); clim([w_min, w_max]); 
    hold on; axis image; axis off;
    if any(ptv_c(:))
        contour(1:W_img, 1:H_img, ptv_c, [0.5 0.5], 'r', 'LineWidth', 2.0); 
    end
    title('CT & Target (PTV)', 'FontSize', 12, 'FontWeight', 'bold');

    % Subplot 2: Demons Reference
    subplot(1,4,2); 
    d_gt_c = orient_func(dose_gt(:,:,slice_z));
    imagesc(1:W_img, 1:H_img, d_gt_c); colormap(gca, 'jet'); clim([0 62]); 
    hold on; axis image; axis off;
    if max(d_gt_c(:)) > 55.0
        contour(1:W_img, 1:H_img, d_gt_c, [60 60], 'y--', 'LineWidth', 1.5); 
    end
    if any(ptv_c(:))
        contour(1:W_img, 1:H_img, ptv_c, [0.5 0.5], 'r-', 'LineWidth', 1.5); 
    end
    title(sprintf('Demons Reference\nD95: %.1f Gy', calc_D95(dose_gt)), 'FontSize', 12);

    % Subplot 3: OLS Baseline
    subplot(1,4,3); 
    d_ols_c = orient_func(dose_ols(:,:,slice_z));
    imagesc(1:W_img, 1:H_img, d_ols_c); colormap(gca, 'jet'); clim([0 62]); 
    hold on; axis image; axis off;
    if max(d_ols_c(:)) > 55.0
        contour(1:W_img, 1:H_img, d_ols_c, [60 60], 'y--', 'LineWidth', 1.5); 
    end
    if any(ptv_c(:))
        contour(1:W_img, 1:H_img, ptv_c, [0.5 0.5], 'r-', 'LineWidth', 1.5); 
    end
    title(sprintf('OLS (Baseline)\nD95: %.1f Gy', calc_D95(dose_ols)), 'FontSize', 12);

    % Subplot 4: KIDE Ours
    subplot(1,4,4); 
    d_kide_c = orient_func(dose_kide(:,:,slice_z));
    imagesc(1:W_img, 1:H_img, d_kide_c); colormap(gca, 'jet'); clim([0 62]); 
    hold on; axis image; axis off;
    if max(d_kide_c(:)) > 55.0
        contour(1:W_img, 1:H_img, d_kide_c, [60 60], 'y--', 'LineWidth', 1.5); 
    end
    if any(ptv_c(:))
        contour(1:W_img, 1:H_img, ptv_c, [0.5 0.5], 'r-', 'LineWidth', 1.5); 
    end
    title(sprintf('KIDE (Ours)\nD95: %.1f Gy', calc_D95(dose_kide)), 'FontSize', 12);

    cb = colorbar('Position', [0.93, 0.15, 0.015, 0.7]);
    cb.Label.String = 'Dose (Gy)'; cb.Label.FontSize = 11; cb.Label.FontWeight = 'bold';
    exportgraphics(fig3, fullfile(result_path, sprintf('Figure3_Case%d_DoseMap_Fixed.png', case_id)), 'Resolution', 300);

    % ---------------------------------------------------------------------
    % 6. Figure 4: DVH
    % ---------------------------------------------------------------------
    fig4 = figure('Color', 'w', 'Position', [200, 200, 500, 420]);
    edges = 0:1:70;
    calc_dvh = @(dose) 100 * cumsum(histcounts(dose(ptv_mask), edges), 'reverse') / sum(ptv_mask(:));

    plot(edges(1:end-1), calc_dvh(plan_dose), 'k--', 'LineWidth', 2.5); hold on;
    plot(edges(1:end-1), calc_dvh(dose_gt), 'Color', [0.2 0.8 0.2], 'LineWidth', 3);
    plot(edges(1:end-1), calc_dvh(dose_ols), 'r-.', 'LineWidth', 2.5);
    plot(edges(1:end-1), calc_dvh(dose_kide), 'b-', 'LineWidth', 3);
    xlabel('Dose (Gy)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Target Volume (%)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Case %d: PTV Dose-Volume Histogram', case_id), 'FontSize', 13);
    legend('Static Plan', 'Demons Reference', 'OLS (Baseline)', 'KIDE (Ours)', 'Location', 'southwest', 'FontSize', 10);
    grid on; axis([0 70 0 105]);
    exportgraphics(fig4, fullfile(result_path, sprintf('Figure4_Case%d_DVH_Fixed.png', case_id)), 'Resolution', 300);
    fprintf('  ✅ Case %02d 剂量图与 DVH 保存成功 (几何完全同轴对齐)。\n', case_id);
    clear maps X Y Z dose_gt dose_ols dose_kide plan_dose DVFs dvf_gt_raw dvf_ols dvf_kide V_train_sampled coeff_V;
end
T=array2table(dose_rows,'VariableNames',{'Case','ModelID','TargetExcursion_mm','D95_Gy','D98_Gy','V20_pct','Gamma33_pct','Gamma22_pct'});
names=["Plan";"Demons reference";"OLS";"KIDE"]; T.Model=names(T.ModelID);
writetable(T,fullfile(result_path,'R2_Dosimetry.csv'));

%% ========================================================================
%  快速 3D Gamma 指标计算
% =========================================================================
function pass_rate = compute_gamma_3d(eval_dose, ref_dose, mask, dist_crit, dose_crit, vs)
    idx = find(mask);
    if isempty(idx), pass_rate = 100.0; return; end
    [H, W, D] = size(ref_dose);
    [I, J, K] = ind2sub([H, W, D], idx);
    rx = ceil(dist_crit / vs(1)); ry = ceil(dist_crit / vs(2)); rz = ceil(dist_crit / vs(3));
    passed = 0; n_pts = length(idx);
    for p = 1:n_pts
        ci = I(p); cj = J(p); ck = K(p);
        d_ref = ref_dose(ci, cj, ck);
        i_min = max(1, ci - ry); i_max = min(H, ci + ry);
        j_min = max(1, cj - rx); j_max = min(W, cj + rx);
        k_min = max(1, ck - rz); k_max = min(D, ck + rz);
        [sub_j, sub_i, sub_k] = meshgrid(j_min:j_max, i_min:i_max, k_min:k_max);
        d2 = ((sub_j - cj)*vs(1)).^2 + ((sub_i - ci)*vs(2)).^2 + ((sub_k - ck)*vs(3)).^2;
        sub_eval = eval_dose(i_min:i_max, j_min:j_max, k_min:k_max);
        gamma_sq = d2 / (dist_crit^2) + ((sub_eval - d_ref).^2) / (dose_crit^2);
        if min(gamma_sq(:)) <= 1.0, passed = passed + 1; end
    end
    pass_rate = (passed / n_pts) * 100.0;
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
