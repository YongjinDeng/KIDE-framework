%% ========================================================================
%  Phase 2: Heterogeneity-Aware Dose Engine
%  新增: 全局等距 Dose Map 排版、V20 肺部毒性指标、60Gy等剂量线
% =========================================================================
clear; clc; close all;

main_path = 'D:/0临床科研/四维剂量重建/data/';
target_cases = [1, 5, 8];

fprintf('==================================================\n');
fprintf('Heterogeneity-Aware Dosimetric Evaluation & DVH\n');
fprintf('==================================================\n');

for case_id = target_cases
    fprintf('Processing Case %d...\n', case_id);
    case_path = fullfile(main_path, sprintf('Case%dPack/', case_id));
    [img_size, voxel_spacing] = get_dirlab_params(case_id);
    
    processed_file = fullfile(case_path, 'Processed', sprintf('Case%d_DVF_Depth.mat', case_id));
    if ~exist(processed_file, 'file'), continue; end
    load(processed_file, 'depth_maps', 'DVFs'); 
    dvf_gt_raw = DVFs{6};
    
    % --- 极度鲁棒的文件搜索 ---
    img_files = dir(fullfile(case_path, 'Images', '*T00*.img'));
    if isempty(img_files)
        fprintf('  ⚠️ CT图像不存在，跳过\n'); continue;
    end
    img_file = fullfile(img_files(1).folder, img_files(1).name);
    
    fid = fopen(img_file, 'r');
    if fid == -1, fprintf('  ⚠️ 无法打开CT图像，跳过\n'); continue; end
    img_T00 = permute(reshape(fread(fid, prod(img_size), 'int16'), img_size), [2, 1, 3]); 
    fclose(fid);
    
    % --- 定位靶区与肺部 (为 V20 做准备) ---
    img_norm = (double(img_T00) - min(double(img_T00(:)))) / (max(double(img_T00(:))) - min(double(img_T00(:))));
    body_mask = img_norm > graythresh(img_norm(:, :, round(img_size(3)/2)));
    for z = 1:img_size(3), body_mask(:,:,z) = imfill(body_mask(:,:,z), 'holes'); end
    CC_body = bwconncomp(body_mask, 26); [~, max_idx] = max(cellfun(@numel, CC_body.PixelIdxList));
    patient_body = false(size(body_mask)); patient_body(CC_body.PixelIdxList{max_idx}) = true;
    
    lung_mask = patient_body & (img_norm < graythresh(img_norm(patient_body)));
    CC_lung = bwconncomp(lung_mask, 26); [~, sorted_idx] = sort(cellfun(@numel, CC_lung.PixelIdxList), 'descend');
    lung_mask_clean = false(size(lung_mask)); 
    for i = 1:min(2, length(sorted_idx)), lung_mask_clean(CC_lung.PixelIdxList{sorted_idx(i)}) = true; end
    
    stats = regionprops(CC_lung, 'Centroid', 'Area'); [~, max_lung_idx] = max([stats.Area]);
    ptv_center = round(stats(max_lung_idx).Centroid); ptv_center(3) = min(ptv_center(3) + round(0.15 * img_size(3)), img_size(3)-5);
    
    [X, Y, Z] = meshgrid(1:img_size(2), 1:img_size(1), 1:img_size(3));
    ptv_radius = 15.0;
    dist_map = sqrt(((X-ptv_center(1))*voxel_spacing(1)).^2 + ((Y-ptv_center(2))*voxel_spacing(2)).^2 + ((Z-ptv_center(3))*voxel_spacing(3)).^2);
    ptv_mask = (dist_map <= ptv_radius) & lung_mask_clean;
    
    % --- 构造高保形 SBRT 剂量分布 ---
    presc_dose = 62.0; 
    density_map = max(0.1, min(1.5, (double(img_T00) + 1000) / 1000));
    eff_dist = dist_map .* (1.0 + 0.15 * (1.0 - density_map)); 
    
    margin = 3.0; steepness = 1.5; 
    plan_dose = presc_dose ./ (1 + exp(steepness * (eff_dist - (ptv_radius + margin))));
    plan_dose(plan_dose > presc_dose) = presc_dose;
    
    % =========================================================================
    % 核心算法还原：完全使用您的原始经典逻辑 (全周期 10 相 PCA)
    % =========================================================================
    D_mat = reshape(depth_maps, [], 10)'; 
    [coeff_S, score_S, ~] = pca(D_mat, 'Economy', true); 
    Z_surf = score_S(:, 1:3);
    
    if prod(img_size) > 1e6, scale_factor = 0.5; else, scale_factor = 1.0; end
    small_size = max(floor(img_size * scale_factor), [1, 1, 1]);
    
    V_mat_sampled = zeros(10, prod(small_size)*3, 'single');
    for i = 1:10
        dvf_s = zeros([small_size, 3], 'single');
        for c = 1:3, dvf_s(:,:,:,c) = imresize3(DVFs{i}(:,:,:,c), small_size, 'linear'); end
        V_mat_sampled(i, :) = dvf_s(:);
    end
    [coeff_V, score_V, latent_V] = pca(V_mat_sampled, 'Economy', true); 
    Y_dvf = score_V(:, 1:3); 
    mean_V = mean(V_mat_sampled, 1);
    
    Z_train = Z_surf(1:5, :); Y_train = Y_dvf(1:5, :); 
    H = Z_train'*Z_train; scale_H = max(abs(H(:))); if scale_H<1e-6, scale_H=1; end
    W_ols = (H + 1e-6*scale_H*eye(3)) \ (Z_train'*Y_train);
    
    penalty_diag = 1 ./ (latent_V(1:3) / max(latent_V(1:3)) + 1e-4); penalty_diag(1) = 0;
    W_kide = (H + 5.0*scale_H*diag(penalty_diag)) \ (Z_train'*Y_train);
    
    % 安全噪声注入 (保留您最初的写法)
    rng(case_id); 
    clean_depth = depth_maps(:,:,6);
    noisy_depth = clean_depth + imgaussfilt(randn(size(clean_depth)), 2.0) * 5.0;
    [r, c] = size(noisy_depth); occ_w = round(min(r, c) * 0.15); cx = round(r/2); cy = round(c/2);
    r_idx = max(1, cx - occ_w) : min(r, cx + occ_w); c_idx = max(1, cy - occ_w) : min(c, cy + occ_w);
    noisy_depth(r_idx, c_idx) = noisy_depth(r_idx, c_idx) + 150.0;
    
    Z_test_noisy = ((noisy_depth(:)' - mean(reshape(depth_maps(:,:,1:5), [], 5)', 1)) * coeff_S); 
    Z_test_noisy = Z_test_noisy(1:3);
    
    Y_ols = Z_test_noisy * W_ols; 
    Y_kide = Z_test_noisy * W_kide;
    
    dvf_ols_s = reshape(mean_V + Y_ols * coeff_V(:, 1:3)', [small_size, 3]); 
    dvf_kide_s = reshape(mean_V + Y_kide * coeff_V(:, 1:3)', [small_size, 3]);
    dvf_ols = zeros([img_size, 3], 'single'); dvf_kide = zeros([img_size, 3], 'single');
    for ch = 1:3
        dvf_ols(:,:,:,ch) = imresize3(dvf_ols_s(:,:,:,ch), img_size, 'linear'); 
        dvf_kide(:,:,:,ch) = imresize3(dvf_kide_s(:,:,:,ch), img_size, 'linear');
    end
    % =========================================================================
    
    % --- 提取动态剂量 ---
    warp_dose = @(DVF) interp3(X, Y, Z, plan_dose, X + DVF(:,:,:,1)/voxel_spacing(1), Y + DVF(:,:,:,2)/voxel_spacing(2), Z + DVF(:,:,:,3)/voxel_spacing(3), 'linear', 0);
    dose_gt   = warp_dose(dvf_gt_raw); dose_ols  = warp_dose(dvf_ols); dose_kide = warp_dose(dvf_kide);
    
    % 评估指标: D95 (覆盖) 和 V20 (肺部毒性)
    calc_D95 = @(d) prctile(d(ptv_mask), 5);
    calc_V20 = @(d) sum(d(lung_mask_clean) >= 20.0) / sum(lung_mask_clean(:)) * 100;
    
    fprintf('  D95%% (靶区覆盖) : Plan=%.1f | GT(True)=%.1f | OLS=%.1f | KIDE=%.1f Gy\n', calc_D95(plan_dose), calc_D95(dose_gt), calc_D95(dose_ols), calc_D95(dose_kide));
    fprintf('  V20  (肺部毒性) : Plan=%.1f%% | GT(True)=%.1f%% | OLS=%.1f%% | KIDE=%.1f%%\n', calc_V20(plan_dose), calc_V20(dose_gt), calc_V20(dose_ols), calc_V20(dose_kide));
    
    % --- 绘图：Figure 3 (绝对完美排版 + 等剂量线) ---
    slice_z = ptv_center(3);
    fig3 = figure('Color', 'w', 'Position', [50, 100, 1500, 350]); % 更修长，适合4图并排
    
    img_c = rot90(double(img_T00(:,:,slice_z)'), -1); ptv_c = rot90(ptv_mask(:,:,slice_z)', -1);
    w_min = prctile(img_c(:), 2); w_max = prctile(img_c(:), 98);
    
    % Subplot 1: CT + PTV
    ax1 = subplot(1,4,1);
    imshow(img_c, [w_min, w_max]); hold on;
    contour(ptv_c, 1, 'r', 'LineWidth', 2.0);
    title('CT & Target (PTV)', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Subplot 2: Ground Truth
    ax2 = subplot(1,4,2);
    d_gt_c = rot90(dose_gt(:,:,slice_z)', -1);
    imagesc(d_gt_c); colormap(gca, 'jet'); caxis([0 62]); hold on; axis image; axis off;
    contour(d_gt_c, [60 60], 'y--', 'LineWidth', 1.5); % 60Gy等剂量线
    contour(ptv_c, 1, 'r-', 'LineWidth', 1.5); % 红色PTV基准
    title(sprintf('Ground Truth\nD95: %.1f Gy', calc_D95(dose_gt)), 'FontSize', 14);
    
    % Subplot 3: OLS
    ax3 = subplot(1,4,3);
    d_ols_c = rot90(dose_ols(:,:,slice_z)', -1);
    imagesc(d_ols_c); colormap(gca, 'jet'); caxis([0 62]); hold on; axis image; axis off;
    contour(d_ols_c, [60 60], 'y--', 'LineWidth', 1.5);
    contour(ptv_c, 1, 'r-', 'LineWidth', 1.5);
    title(sprintf('OLS (Baseline)\nD95: %.1f Gy', calc_D95(dose_ols)), 'FontSize', 14);
    
    % Subplot 4: KIDE
    ax4 = subplot(1,4,4);
    d_kide_c = rot90(dose_kide(:,:,slice_z)', -1);
    imagesc(d_kide_c); colormap(gca, 'jet'); caxis([0 62]); hold on; axis image; axis off;
    contour(d_kide_c, [60 60], 'y--', 'LineWidth', 1.5);
    contour(ptv_c, 1, 'r-', 'LineWidth', 1.5);
    title(sprintf('KIDE (Ours)\nD95: %.1f Gy', calc_D95(dose_kide)), 'FontSize', 14);
    
    % 添加统一的独立 Colorbar
    cb = colorbar('Position', [0.93, 0.15, 0.015, 0.7]);
    cb.Label.String = 'Dose (Gy)';
    cb.Label.FontSize = 12;
    cb.Label.FontWeight = 'bold';
    
    exportgraphics(fig3, fullfile(main_path, sprintf('Figure3_Case%d_DoseMap.png', case_id)), 'Resolution', 300);
    
    % --- 绘图：Figure 4 (DVH 曲线精美版) ---
    fig4 = figure('Color', 'w', 'Position', [200, 200, 500, 450]);
    edges = 0:1:70;
    calc_dvh = @(dose) 100 * cumsum(histcounts(dose(ptv_mask), edges), 'reverse') / sum(ptv_mask(:));
    
    plot(edges(1:end-1), calc_dvh(plan_dose), 'k--', 'LineWidth', 2.5); hold on;
    plot(edges(1:end-1), calc_dvh(dose_gt), 'Color', [0.2 0.8 0.2], 'LineWidth', 3);
    plot(edges(1:end-1), calc_dvh(dose_ols), 'r-.', 'LineWidth', 2.5);
    plot(edges(1:end-1), calc_dvh(dose_kide), 'b-', 'LineWidth', 3);
    
    xlabel('Dose (Gy)', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Target Volume (%)', 'FontSize', 13, 'FontWeight', 'bold');
    title(sprintf('Case %d: PTV Dose-Volume Histogram', case_id), 'FontSize', 15);
    legend('Static Plan', 'Ground Truth', 'OLS (False Tracking)', 'KIDE (Ours)', ...
        'Location', 'southwest', 'FontSize', 11);
    grid on; ax = gca; ax.GridAlpha = 0.4; ax.LineWidth = 1.2;
    axis([0 70 0 105]);
    exportgraphics(fig4, fullfile(main_path, sprintf('Figure4_Case%d_DVH.png', case_id)), 'Resolution', 300);
    
    fprintf('  ✅ Figure 3 (Dose Map) & Figure 4 (DVH) 已保存\n\n');
end
fprintf('All Dosimetric Evaluations Completed.\n');

function [img_size, vs] = get_dirlab_params(case_id)
    if case_id<=5, img_size=[256, 256, 94]; vs=[0.97, 0.97, 2.5]; else, img_size=[512, 512, 128]; vs=[0.97, 0.97, 2.5]; end
    switch case_id
        case 1, img_size(3)=94; case 2, img_size(3)=112; vs(1:2)=[1.16, 1.16];
        case 3, img_size(3)=104; vs(1:2)=[1.15, 1.15]; case 4, img_size(3)=99; vs(1:2)=[1.13, 1.13];
        case 5, img_size(3)=106; vs(1:2)=[1.10, 1.10]; case 7, img_size(3)=136; case 10, img_size(3)=120;
    end
end
