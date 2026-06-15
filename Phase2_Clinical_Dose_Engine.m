%% ========================================================================
%  Phase 2: Clinical Anatomy-Aware Dosimetric Evaluation
%  For journal: Radiotherapy and Oncology
%  Description: Evaluates the dosimetric impact of intra-fraction motion
%               using a Sigmoid-based SBRT pseudo-dose engine.
% =========================================================================
clear; clc; close all;

main_path = 'D:/0临床科研/四维剂量重建/data/';
target_cases = [1, 5, 8];
all_dose_results = [];

fprintf('==================================================\n');
fprintf('Dosimetric Evaluation\n');
fprintf('==================================================\n');

for case_id = target_cases
    fprintf('Processing Case %d...\n', case_id);
    case_path = fullfile(main_path, sprintf('Case%dPack/', case_id));
    [img_size, voxel_spacing] = get_dirlab_params(case_id);
    
    % 1. Load Preprocessed Data
    processed_file = fullfile(case_path, 'Processed', sprintf('Case%d_DVF_Depth.mat', case_id));
    if ~exist(processed_file, 'file'), continue; end
    load(processed_file, 'depth_maps', 'DVFs');
    
    img_file = '';
    possible_names = {'case%d_T00.img', 'case%d_T00_s.img', 'case%d_T00-ssm.img'};
    for i = 1:length(possible_names)
        tmp = fullfile(case_path, 'Images', sprintf(possible_names{i}, case_id));
        if exist(tmp, 'file'), img_file = tmp; break; end
    end
    if isempty(img_file), continue; end
    
    fid = fopen(img_file, 'r'); 
    img_T00 = reshape(fread(fid, prod(img_size), 'int16'), img_size); 
    fclose(fid);
    img_T00 = permute(img_T00, [2, 1, 3]);
    
    % 2. Adaptive Lung & PTV Segmentation
    img_norm = (double(img_T00) - min(double(img_T00(:)))) / (max(double(img_T00(:))) - min(double(img_T00(:))));
    
    body_level = graythresh(img_norm(:, :, round(img_size(3)/2)));
    body_mask = img_norm > body_level;
    for z = 1:img_size(3), body_mask(:,:,z) = imfill(body_mask(:,:,z), 'holes'); end
    CC_body = bwconncomp(body_mask, 26);
    [~, max_idx] = max(cellfun(@numel, CC_body.PixelIdxList));
    patient_body = false(size(body_mask)); patient_body(CC_body.PixelIdxList{max_idx}) = true;
    
    lung_level = graythresh(img_norm(patient_body)); 
    lung_mask = patient_body & (img_norm < lung_level);
    
    CC_lung = bwconncomp(lung_mask, 26);
    numPixels = cellfun(@numel, CC_lung.PixelIdxList);
    [~, sorted_idx] = sort(numPixels, 'descend');
    lung_mask_clean = false(size(lung_mask));
    for i = 1:min(2, length(sorted_idx))
        lung_mask_clean(CC_lung.PixelIdxList{sorted_idx(i)}) = true;
    end
    
    stats = regionprops(CC_lung, 'Centroid', 'Area'); 
    [~, max_lung_idx] = max([stats.Area]);
    ptv_center = round(stats(max_lung_idx).Centroid); 
    ptv_center(3) = min(ptv_center(3) + round(0.15 * img_size(3)), img_size(3)-5); 
    
    [X, Y, Z] = meshgrid(1:img_size(2), 1:img_size(1), 1:img_size(3));
    ptv_radius = 15.0;
    dist_sq = ((X - ptv_center(1))*voxel_spacing(1)).^2 + ((Y - ptv_center(2))*voxel_spacing(2)).^2 + ((Z - ptv_center(3))*voxel_spacing(3)).^2;
    dist_map = sqrt(dist_sq);
    ptv_mask = (dist_map <= ptv_radius) & lung_mask_clean;
    
    % 3. Sigmoid-based SBRT Analytical Dosimetric Surrogate
    presc_dose = 62.0;
    steepness = 2.0;
    plan_dose = presc_dose ./ (1 + exp(steepness * (dist_map - (ptv_radius + 4))));
    
    % 4. DVF Prediction
    D_mat = reshape(depth_maps, [], 10)'; [~, score_S, ~] = pca(D_mat, 'Economy', true); Z_surf = score_S(:, 1:3);
    V_mat = zeros(10, numel(DVFs{1}), 'single'); for i=1:10, V_mat(i,:)=DVFs{i}(:); end
    [coeff_V, score_V, latent_V] = pca(V_mat, 'Economy', true); Y_dvf = score_V(:, 1:3); mean_V = mean(V_mat, 1);
    
    Z_train = Z_surf(1:5, :); Y_train = Y_dvf(1:5, :);
    H = Z_train'*Z_train; scale_H = max(abs(H(:))); if scale_H < 1e-6, scale_H = 1; end
    
    W_ols = (H + 1e-6*scale_H*eye(3)) \ (Z_train'*Y_train);
    penalty_diag = 1 ./ (latent_V(1:3) / max(latent_V(1:3)) + 1e-4); penalty_diag(1) = 0;
    W_pide = (H + 5.0*scale_H*diag(penalty_diag)) \ (Z_train'*Y_train);
    
    Z_test_clean = Z_surf(6, :);
    rng(case_id); sensor_noise = randn(1, 3) .* std(Z_train) .* [0.1, 2.0, 5.0]; 
    Z_test_noisy = Z_test_clean + sensor_noise;
    
    Y_pred_ols = Z_test_noisy * W_ols; Y_pred_pide = Z_test_noisy * W_pide;
    dvf_ols = reshape(mean_V + Y_pred_ols * coeff_V(:, 1:3)', [img_size, 3]);
    dvf_pide = reshape(mean_V + Y_pred_pide * coeff_V(:, 1:3)', [img_size, 3]);
    dvf_gt = DVFs{6};
    
    % 5. Dose Warping
    shift_gt_x = dvf_gt(:,:,:,1)/voxel_spacing(1); shift_gt_y = dvf_gt(:,:,:,2)/voxel_spacing(2); shift_gt_z = dvf_gt(:,:,:,3)/voxel_spacing(3);
    shift_ols_x = dvf_ols(:,:,:,1)/voxel_spacing(1); shift_ols_y = dvf_ols(:,:,:,2)/voxel_spacing(2); shift_ols_z = dvf_ols(:,:,:,3)/voxel_spacing(3);
    shift_pide_x = dvf_pide(:,:,:,1)/voxel_spacing(1); shift_pide_y = dvf_pide(:,:,:,2)/voxel_spacing(2); shift_pide_z = dvf_pide(:,:,:,3)/voxel_spacing(3);
    
    dose_gt   = interp3(X, Y, Z, plan_dose, X + shift_gt_x, Y + shift_gt_y, Z + shift_gt_z, 'linear', 0);
    dose_ols  = interp3(X, Y, Z, plan_dose, X + shift_ols_x, Y + shift_ols_y, Z + shift_ols_z, 'linear', 0);
    dose_pide = interp3(X, Y, Z, plan_dose, X + shift_pide_x, Y + shift_pide_y, Z + shift_pide_z, 'linear', 0);
    
    % 6. Extract Dosimetric Indices
    if sum(ptv_mask(:)) > 0
        D95_plan = prctile(plan_dose(ptv_mask), 5); D95_gt = prctile(dose_gt(ptv_mask), 5);
        D95_ols = prctile(dose_ols(ptv_mask), 5); D95_pide = prctile(dose_pide(ptv_mask), 5);
        
        fprintf('  D95%%: Plan=%.1f | GT=%.1f | OLS=%.1f | PIDE=%.1f Gy\n', D95_plan, D95_gt, D95_ols, D95_pide);
        
        % 7. Visualization (High Quality for Publication)
        slice_z = ptv_center(3);
        fig = figure('Color', 'w', 'Position', [100, 100, 1600, 400]);
        
        % 修正朝向
        img_slice = double(img_T00(:,:,slice_z)');
        ptv_slice = ptv_mask(:,:,slice_z)';
        
        img_corrected = rot90(img_slice, -1);
        ptv_corrected = rot90(ptv_slice, -1);
        dose_gt_corrected = rot90(dose_gt(:,:,slice_z)', -1);
        dose_ols_corrected = rot90(dose_ols(:,:,slice_z)', -1);
        dose_pide_corrected = rot90(dose_pide(:,:,slice_z)', -1);
        
        % --- 关键修复 1: 自动计算最佳的CT窗宽窗位 ---
        w_min = prctile(img_corrected(:), 2); % 剔除背景极小值
        w_max = prctile(img_corrected(:), 98); % 剔除极高密度骨骼/金属
        if w_max <= w_min, w_max = w_min + 1; end
        
        % 子图1: CT + PTV
        subplot(1,4,1);
        imshow(img_corrected, [w_min, w_max]); hold on;
        contour(ptv_corrected, 1, 'r', 'LineWidth', 2.0); % 加粗 PTV
        title('CT & Target (PTV)', 'FontSize', 12);
        axis image; % --- 关键修复 2: 保持物理比例 ---
        axis off;
        
        % 子图2: Ground Truth Dose
        subplot(1,4,2);
        imagesc(dose_gt_corrected);
        colormap(gca, 'jet');
        caxis([0 62]);
        colorbar;
        title(sprintf('Ground Truth Dose\nD95: %.1f Gy', D95_gt), 'FontSize', 12);
        axis image; axis off;
        
        % 子图3: OLS Prediction
        subplot(1,4,3);
        imagesc(dose_ols_corrected);
        colormap(gca, 'jet');
        caxis([0 62]);
        colorbar;
        title(sprintf('OLS Prediction\nD95: %.1f Gy', D95_ols), 'FontSize', 12);
        axis image; axis off;
        
        % 子图4: PIDE Prediction
        subplot(1,4,4);
        imagesc(dose_pide_corrected);
        colormap(gca, 'jet');
        caxis([0 62]);
        colorbar;
        title(sprintf('PIDE Prediction\nD95: %.1f Gy', D95_pide), 'FontSize', 12);
        axis image; axis off;
        
        sgtitle(sprintf('Case %d: Dosimetric Impact Evaluated via Analytical Surrogate', case_id), ...
            'FontSize', 14, 'FontWeight', 'bold');
        
        % 保存
        exportgraphics(gcf, fullfile(main_path, sprintf('Figure3_Case%d_Dose.png', case_id)), 'Resolution', 300);
        exportgraphics(gcf, fullfile(main_path, sprintf('Figure3_Case%d_Dose.pdf', case_id)), 'ContentType', 'vector');
        fprintf('  ✅ Figure 3 (Case %d) 已保存\n', case_id);
    end
end
fprintf('Evaluation Completed.\n');

%% 辅助函数
function [img_size, voxel_spacing] = get_dirlab_params(case_id)
    if case_id <= 5
        img_size = [256, 256, 94];
        voxel_spacing = [0.97, 0.97, 2.5];
    else
        img_size = [512, 512, 128];
        voxel_spacing = [0.97, 0.97, 2.5];
    end
    switch case_id
        case 1, img_size(3)=94;
        case 2, img_size(3)=112; voxel_spacing(1:2)=[1.16, 1.16];
        case 3, img_size(3)=104; voxel_spacing(1:2)=[1.15, 1.15];
        case 4, img_size(3)=99; voxel_spacing(1:2)=[1.13, 1.13];
        case 5, img_size(3)=106; voxel_spacing(1:2)=[1.10, 1.10];
        case 7, img_size(3)=136;
        case 10, img_size(3)=120;
    end
end