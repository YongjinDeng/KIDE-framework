%% ========================================================================
%  Batch Process: 批量处理 DIR-Lab Case 1-10
%  For journal: Radiotherapy and Oncology
% =========================================================================
clear; clc; close all;

main_path = 'D:/0临床科研/四维剂量重建/data/';
output_summary = [];

% 收集全部 3000 个标志点的误差
all_initial_err = []; all_base_err = []; all_pide_err = [];

fprintf('==================================================\n');
fprintf('批量处理 DIR-Lab 10 Cases\n');
fprintf('==================================================\n');

for case_id = 1:10
    fprintf('\n==================================================\n');
    fprintf('处理 Case %d / 10\n', case_id);
    
    case_path = fullfile(main_path, sprintf('Case%dPack/', case_id));
    images_path = fullfile(case_path, 'Images');
    landmarks_path = fullfile(case_path, 'ExtremePhases');
    output_dir = fullfile(case_path, 'Processed');
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    
    % 获取当前 Case 的动态参数
    [img_size, voxel_spacing] = get_dirlab_params(case_id);
    phases = {'00', '10', '20', '30', '40', '50', '60', '70', '80', '90'};
    
    processed_file = fullfile(output_dir, sprintf('Case%d_DVF_Depth.mat', case_id));
    
    % 数据准备：加载或计算 Depth & DVF
    if exist(processed_file, 'file')
        fprintf('  [加载缓存] 发现已处理数据，直接加载...\n');
        load(processed_file, 'depth_maps', 'DVFs', 'pts_T00', 'pts_T50');
    else
        fprintf('  [耗时预警] 未发现缓存，开始提取体表并计算 3D 形变场...\n');
        
        % 读取标志点
        if case_id <= 5
            pts_T00 = load(fullfile(landmarks_path, sprintf('Case%d_300_T00_xyz.txt', case_id)));
            pts_T50 = load(fullfile(landmarks_path, sprintf('Case%d_300_T50_xyz.txt', case_id)));
        else
            pts_T00 = load(fullfile(landmarks_path, sprintf('case%d_dirLab300_T00_xyz.txt', case_id)));
            pts_T50 = load(fullfile(landmarks_path, sprintf('case%d_dirLab300_T50_xyz.txt', case_id)));
        end
        
        % 读取图像并提取深度
        images_all = zeros([img_size, 10], 'int16');
        depth_maps = zeros(img_size(1), img_size(3), 10);
        DVFs = cell(10, 1);
        
        for p = 1:10
            % 处理文件名规律不一致的问题
            if case_id == 1
                img_name = sprintf('case%d_T%s_s.img', case_id, phases{p});
            elseif case_id <= 5
                img_name = sprintf('case%d_T%s-ssm.img', case_id, phases{p});
            else
                img_name = sprintf('case%d_T%s.img', case_id, phases{p});
            end
            
            fid = fopen(fullfile(images_path, img_name), 'r');
            img = reshape(fread(fid, prod(img_size), 'int16'), img_size);
            fclose(fid);
            img = permute(img, [2, 1, 3]);
            images_all(:,:,:,p) = img;
            
            % 自适应体表提取
            img_norm = (double(img) - min(double(img(:)))) / (max(double(img(:))) - min(double(img(:))));
            level = graythresh(img_norm(:, :, round(img_size(3)/2)));
            body_mask = img_norm > level;
            for z = 1:img_size(3), body_mask(:,:,z) = imfill(body_mask(:,:,z), 'holes'); end
            CC = bwconncomp(body_mask, 26);
            numPixels = cellfun(@numel, CC.PixelIdxList); [~, max_idx] = max(numPixels);
            patient_body = false(size(body_mask)); patient_body(CC.PixelIdxList{max_idx}) = true;
            
            curr_depth = NaN(img_size(1), img_size(3));
            for x = 1:img_size(1)
                for z = 1:img_size(3)
                    y_skin = find(patient_body(x, :, z), 1, 'first');
                    if ~isempty(y_skin), curr_depth(x, z) = y_skin * voxel_spacing(2); end
                end
            end
            curr_depth(1:round(img_size(1)*0.2), :) = NaN; curr_depth(round(img_size(1)*0.8):end, :) = NaN;
            curr_depth = medfilt2(fillmissing(curr_depth, 'nearest'), [5, 5]);
            depth_maps(:,:,p) = img_size(2)*voxel_spacing(2) - curr_depth;
        end
        
        % 计算真实 DVF
        ref_img = double(images_all(:,:,:,1)); ref_img = (ref_img - min(ref_img(:)))/(max(ref_img(:)) - min(ref_img(:)));
        DVFs{1} = zeros([img_size, 3], 'single');
        for p = 2:10
            fprintf('    计算配准 DVF T%s -> T00...\n', phases{p});
            mov_img = double(images_all(:,:,:,p)); mov_img = (mov_img - min(mov_img(:)))/(max(mov_img(:)) - min(mov_img(:)));
            [DVF, ~] = imregdemons(mov_img, ref_img, [50 25 10], 'AccumulatedFieldSmoothing', 1.5, 'DisplayWaitbar', false);
            DVFs{p} = single(DVF);
        end
        save(processed_file, 'depth_maps', 'DVFs', 'pts_T00', 'pts_T50', '-v7.3');
        fprintf('  ✅ 数据处理与缓存完成\n');
    end
    
    % PCA 与模型训练
    fprintf('  PCA 降维与模型训练...\n');
    D_mat = reshape(depth_maps, [], 10)';
    [~, score_S, ~] = pca(D_mat, 'Economy', true);
    k_feat = min(3, size(score_S, 2));
    Z_surf = score_S(:, 1:k_feat);
    
    V_mat = zeros(10, numel(DVFs{1}), 'single');
    for i = 1:10, V_mat(i, :) = DVFs{i}(:); end
    [coeff_V, score_V, latent_V] = pca(V_mat, 'Economy', true);
    Y_dvf = score_V(:, 1:k_feat); mean_V = mean(V_mat, 1);
    
    Z_train = Z_surf; Y_train = Y_dvf;
    H = Z_train' * Z_train; scale_H = max(abs(H(:))); if scale_H < 1e-6, scale_H = 1; end
    
    W_base = (H + 1e-5 * scale_H * eye(k_feat)) \ (Z_train' * Y_train);
    
    alpha_physics = 5.0;
    penalty_diag = 1 ./ (latent_V(1:k_feat) / max(latent_V(1:k_feat)) + 1e-4);
    penalty_diag(1) = 0;
    W_pide = (H + alpha_physics * scale_H * diag(penalty_diag)) \ (Z_train' * Y_train);
    
    % 噪声注入与测试
    Z_test_clean = Z_surf(6, :); 
    rng(case_id); 
    noise = randn(1, k_feat) .* std(Z_train) .* [0.1, 2.0, 5.0]; 
    Z_test_noisy = Z_test_clean + noise;
    
    Y_pred_base = Z_test_noisy * W_base;
    Y_pred_pide = Z_test_noisy * W_pide;
    
    % 重建与 TRE 计算
    dvf_base_vec = mean_V + Y_pred_base * coeff_V(:, 1:k_feat)';
    dvf_pide_vec = mean_V + Y_pred_pide * coeff_V(:, 1:k_feat)';
    DVF_base = reshape(dvf_base_vec, [img_size, 3]);
    DVF_pide = reshape(dvf_pide_vec, [img_size, 3]);
    
    [Xm, Ym, Zm] = meshgrid(1:img_size(2), 1:img_size(1), 1:img_size(3));
    
    dx_b = interp3(Xm, Ym, Zm, double(DVF_base(:,:,:,1)), pts_T00(:,1), pts_T00(:,2), pts_T00(:,3), 'linear', 0);
    dy_b = interp3(Xm, Ym, Zm, double(DVF_base(:,:,:,2)), pts_T00(:,1), pts_T00(:,2), pts_T00(:,3), 'linear', 0);
    dz_b = interp3(Xm, Ym, Zm, double(DVF_base(:,:,:,3)), pts_T00(:,1), pts_T00(:,2), pts_T00(:,3), 'linear', 0);
    
    dx_p = interp3(Xm, Ym, Zm, double(DVF_pide(:,:,:,1)), pts_T00(:,1), pts_T00(:,2), pts_T00(:,3), 'linear', 0);
    dy_p = interp3(Xm, Ym, Zm, double(DVF_pide(:,:,:,2)), pts_T00(:,1), pts_T00(:,2), pts_T00(:,3), 'linear', 0);
    dz_p = interp3(Xm, Ym, Zm, double(DVF_pide(:,:,:,3)), pts_T00(:,1), pts_T00(:,2), pts_T00(:,3), 'linear', 0);
    
    err_initial = sqrt(sum(((pts_T50 - pts_T00) .* voxel_spacing).^2, 2));
    err_base = sqrt(sum((((pts_T00 + [dx_b, dy_b, dz_b]) - pts_T50) .* voxel_spacing).^2, 2));
    err_pide = sqrt(sum((((pts_T00 + [dx_p, dy_p, dz_p]) - pts_T50) .* voxel_spacing).^2, 2));
    
    % Jacobian 撕裂率
    [~, dy_dx, ~] = gradient(DVF_pide(:,:,:,2), voxel_spacing(1), voxel_spacing(2), voxel_spacing(3));
    jac_fold_pide = sum(dy_dx(:) <= -1) / numel(dy_dx) * 100;
    [~, dy_dx_b, ~] = gradient(DVF_base(:,:,:,2), voxel_spacing(1), voxel_spacing(2), voxel_spacing(3));
    jac_fold_base = sum(dy_dx_b(:) <= -1) / numel(dy_dx_b) * 100;
    
    % 收集数据
    all_initial_err = [all_initial_err; err_initial];
    all_base_err = [all_base_err; err_base];
    all_pide_err = [all_pide_err; err_pide];
    
    output_summary = [output_summary; struct('case_id', case_id, 'init', mean(err_initial), ...
        'base', mean(err_base), 'pide', mean(err_pide), 'jac_b', jac_fold_base, 'jac_p', jac_fold_pide)];
    
    fprintf('  ✅ 初始: %.2f mm | OLS: %.2f mm | PIDE: %.2f mm (改善 %.1f%%)\n', ...
        mean(err_initial), mean(err_base), mean(err_pide), (mean(err_base)-mean(err_pide))/mean(err_base)*100);
end

%% 终极大汇总与可视化
fprintf('\n==================================================\n');
fprintf('最终战报：全队列 10 例患者 (总计 3000 标志点)\n');
fprintf('  平均初始误差 : %.2f ± %.2f mm\n', mean(all_initial_err), std(all_initial_err));
fprintf('  OLS TRE      : %.2f ± %.2f mm\n', mean(all_base_err), std(all_base_err));
fprintf('  PIDE TRE     : %.2f ± %.2f mm (改善 %.1f%%)\n', ...
    mean(all_pide_err), std(all_pide_err), ...
    (mean(all_base_err)-mean(all_pide_err))/mean(all_base_err)*100);
fprintf('==================================================\n');

% 统计显著性
[~, p_val] = ttest(all_pide_err, all_base_err);
fprintf('  配对 t-test: p = %.2e\n', p_val);

% 临床成功率
success_ols = mean(all_base_err < 2.0) * 100;
success_pide = mean(all_pide_err < 2.0) * 100;
fprintf('  临床成功率 (TRE < 2mm): OLS = %.1f%%, PIDE = %.1f%%\n', success_ols, success_pide);
fprintf('==================================================\n');

% 生成 Figure 2 (箱线图)
fig1 = figure('Color', 'w', 'Position', [100, 100, 550, 500]);
boxplot([all_initial_err, all_base_err, all_pide_err], ...
    'Labels', {'Initial', 'OLS', 'PIDE'}, ...
    'Colors', 'k', 'Widths', 0.6, 'Symbol', '');
ylabel('TRE (mm)', 'FontSize', 12, 'FontWeight', 'bold');
title('Target Registration Error (3000 Landmarks)', 'FontSize', 14, 'FontWeight', 'bold');
grid on; hold on;
yline(2.0, 'r--', 'Clinical Threshold (2mm)', 'LineWidth', 1.5);
ylim([0, max([mean(all_base_err)+2*std(all_base_err), 10])]);
set(gca, 'FontSize', 11);
exportgraphics(fig1, fullfile(main_path, 'Figure2_TRE.png'), 'Resolution', 300);
exportgraphics(fig1, fullfile(main_path, 'Figure2_TRE.pdf'), 'ContentType', 'vector');
fprintf('✅ Figure 2 已保存\n');

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