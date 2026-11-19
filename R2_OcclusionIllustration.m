%% R2: quantify the physical patch and show CT-derived depth examples.
clear; clc; close all;
main_path='E:\D盘数据转移\data四维剂量重建\DIR-Lab\';
result_path='D:\0临床科研\四维剂量重建(给喜雪改投版本)\result\R2';
if ~exist(result_path,'dir'), mkdir(result_path); end
rows=nan(10,12); reps=[1 5 8];
f=figure('Color','w','Position',[50 50 1450 900]); tl=tiledlayout(f,3,3,'TileSpacing','compact','Padding','compact');
for case_id=1:10
    fprintf('Occlusion coverage Case %02d/10\n',case_id);
    case_path=fullfile(main_path,sprintf('Case%dPack',case_id));
    [img_size,vs]=R2_GetDIRLabParams(case_id);
    S=load(fullfile(case_path,'Processed',sprintf('Case%d_DVF_Depth.mat',case_id)),'depth_maps');
    clean=S.depth_maps(:,:,6); [noisy,patch,info]=R2_PerturbDepth(clean,vs,case_id+100);
    % Reproduce the original pre-fill surface support from the T50 CT.
    files=dir(fullfile(case_path,'Images','*T50*.img'));
    assert(numel(files)==1,'Expected one T50 CT in %s',case_path);
    fid=fopen(fullfile(files(1).folder,files(1).name),'r'); assert(fid>=0);
    raw=fread(fid,prod(img_size),'int16=>int16'); fclose(fid);
    assert(numel(raw)==prod(img_size),'Unexpected raw CT length');
    img=permute(reshape(raw,img_size),[2 1 3]); clear raw;
    I=double(img); I=(I-min(I(:)))/(max(I(:))-min(I(:))); clear img;
    body=I>graythresh(I(:,:,round(img_size(3)/2))); clear I;
    for z=1:img_size(3), body(:,:,z)=imfill(body(:,:,z),'holes'); end
    CC=bwconncomp(body,26); [~,idx]=max(cellfun(@numel,CC.PixelIdxList));
    patient=false(size(body)); patient(CC.PixelIdxList{idx})=true; clear body CC;
    support=squeeze(any(patient,2)); clear patient;
    central=support; central(1:round(img_size(1)*0.2),:)=false;
    central(round(img_size(1)*0.8):end,:)=false;
    assert(nnz(support)>0 && nnz(central)>0);
    rows(case_id,:)=[case_id,info(1:3),100*nnz(patch&support)/nnz(support), ...
        100*nnz(patch&central)/nnz(central),nnz(patch),nnz(support),nnz(central),nnz(patch&central),info(4:5)];
    if ismember(case_id,reps)
        rr=find(reps==case_id); x=(0:size(clean,1)-1)*vs(1); z=(0:size(clean,2)-1)*vs(3);
        lim=[min(clean(:)),max(clean(:))]; if lim(2)<=lim(1),lim(2)=lim(1)+1;end
        maps={clean,noisy,noisy-clean}; titles={'Clean depth','Corrupted depth','Depth change'};
        for j=1:3
            ax=nexttile(tl,(rr-1)*3+j); imagesc(ax,x,z,maps{j}'); axis(ax,'image'); hold(ax,'on');
            set(ax,'YDir','normal'); colormap(ax,parula); colorbar(ax);
            if j<3, clim(ax,lim); else, clim(ax,[-10 90]); end
            contour(ax,x,z,double(patch'),[.5 .5],'r-','LineWidth',1.7);
            contour(ax,x,z,double(central'),[.5 .5],'w--','LineWidth',.8);
            title(ax,sprintf('Case %d: %s',case_id,titles{j})); xlabel(ax,'Transverse (mm)'); ylabel(ax,'Slice direction (mm)');
        end
    end
end
T=array2table(rows,'VariableNames',{'Case','PatchTransverse_mm','PatchSI_mm','MapGrid_pct','ProjectedBody_pct','RetainedProjectedBody_pct','PatchPixels','BodyProjectedPixels','RetainedBodyPixels','OverlapRetainedPixels','PatchStartRow','PatchStartCol'});
disp(T); writetable(T,fullfile(result_path,'R2_OcclusionCoverage.csv'));
exportgraphics(f,fullfile(result_path,'FigureS2_Occlusion.png'),'Resolution',300);
