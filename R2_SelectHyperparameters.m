%% R2: training-only CV; no T50 data or landmarks used for selection.
clear; clc; close all;
main_path='E:\D盘数据转移\data四维剂量重建\DIR-Lab\';
result_path='D:\0临床科研\四维剂量重建(给喜雪改投版本)\result\R2';
if ~exist(result_path,'dir'), mkdir(result_path); end
alpha_candidates=[0.5 1 2 3 4 5 6 8 10];
kappa_candidates=[0.1 0.25 0.5 1];
% Four folds: hold out T10/T20/T30/T40 individually; always retain T00.
% Shared parameter pair selected by equally weighted case/fold mean RMS.
fold_rms=nan(10,4,numel(alpha_candidates),numel(kappa_candidates));
for case_id=1:10
    fprintf('Training CV Case %02d/10\n',case_id);
    f=fullfile(main_path,sprintf('Case%dPack',case_id),'Processed',sprintf('Case%d_DVF_Depth.mat',case_id));
    assert(isfile(f),'Missing cache %s',f);
    % Load only the two cached arrays; index only phases 1:5 below.
    S=load(f,'depth_maps','DVFs');
    [img_size,vs]=R2_GetDIRLabParams(case_id);
    assert(isequal(size(S.DVFs{1}),[img_size 3]));
    small_size=max(floor(img_size*0.5),[1 1 1]);
    D=reshape(S.depth_maps(:,:,1:5),[],5)';
    N=prod(small_size); V=zeros(5,3*N,'single');
    for p=1:5
        for c=1:3
            v=imresize3(S.DVFs{p}(:,:,:,c),small_size,'linear');
            V(p,(c-1)*N+(1:N))=v(:)';
        end
    end
    clear S;
    for p=2:5
        train=setdiff(1:5,p); fold=p-1;
        Dtr=D(train,:); muD=mean(Dtr,1);
        [Cs,Z]=pca(Dtr,'Economy',true); Cs=Cs(:,1:3); Z=double(Z(:,1:3));
        Vtr=V(train,:); muV=mean(Vtr,1);
        [Cv,Y,L]=pca(Vtr,'Economy',true); Cv=Cv(:,1:3); Y=double(Y(:,1:3)); L=double(L(1:3));
        assert(all(isfinite(L)) && L(1)>0,'Invalid DVF spectrum');
        depth=reshape(D(p,:),img_size(1),img_size(3));
        noisy=R2_PerturbDepth(depth,vs,1000+10*case_id+p);
        z=(noisy(:)'-muD)*Cs;
        H=Z'*Z; s=max(abs(H(:))); if s<1e-6,s=1;end
        % Exact quadratic RMS in physical mm on sampled grid, avoiding full
        % voxel reconstructions for each candidate. e=muV-Vheld; B=Cv.
        G=zeros(3); b=zeros(3,1); a=0;
        for c=1:3
            idx=(c-1)*N+(1:N);
            B=double(Cv(idx,:)); e=double(muV(idx)-V(p,idx))';
            G=G+vs(c)^2*(B'*B); b=b+vs(c)^2*(B'*e); a=a+vs(c)^2*(e'*e);
        end
        clear B e Cv Vtr;
        for ia=1:numel(alpha_candidates)
            for ik=1:numel(kappa_candidates)
                P=(L(1)-L)./(L+kappa_candidates(ik)*L(1)+1e-6); P(1)=0;
                W=(H+alpha_candidates(ia)*s*diag(P))\(Z'*Y);
                y=(z*W)';
                fold_rms(case_id,fold,ia,ik)=sqrt(max(0,(a+2*y'*b+y'*G*y)/N));
            end
        end
    end
    clear V D;
end
assert(all(isfinite(fold_rms(:))),'Incomplete training CV');
cv_mean=reshape(mean(mean(fold_rms,1),2),numel(alpha_candidates),numel(kappa_candidates));
% Minimum mean RMS; ties follow ascending kappa then ascending alpha.
[min_rms,idx]=min(cv_mean(:)); [ia,ik]=ind2sub(size(cv_mean),idx);
alpha_selected=alpha_candidates(ia); kappa_selected=kappa_candidates(ik);
[AA,KK]=ndgrid(alpha_candidates,kappa_candidates);
writetable(table(AA(:),KK(:),cv_mean(:),'VariableNames',{'Alpha','Kappa','MeanCV_RMS_mm'}),fullfile(result_path,'R2_TrainingCV.csv'));
save(fullfile(result_path,'R2_TrainingCV.mat'),'alpha_candidates','kappa_candidates','fold_rms','cv_mean','alpha_selected','kappa_selected','min_rms');
fprintf('Training-only selection: alpha=%g, kappa=%g, CV RMS=%.4f mm\n',alpha_selected,kappa_selected,min_rms);
f=figure('Color','w','Position',[100 100 750 500]); hold on;
for j=1:numel(kappa_candidates)
    semilogx(alpha_candidates,cv_mean(:,j),'-o','DisplayName',sprintf('\\kappa = %g',kappa_candidates(j)),'LineWidth',1.5);
end
plot(alpha_selected,min_rms,'kp','MarkerSize',13,'MarkerFaceColor','k','DisplayName','Selected pair');
xlabel('Regularization strength (\alpha)'); ylabel('Mean training-CV DVF RMS (mm)');
set(gca,'XScale','log');
title('Training-only cross-validation (10 cases, 4 folds per case)'); grid on; legend('Location','best');
exportgraphics(f,fullfile(result_path,'Figure2a_TrainingCV.png'),'Resolution',300);
