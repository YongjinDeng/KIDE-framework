function [alpha,kappa] = R2_LoadParameters(result_path)
p = fullfile(result_path,'R2_TrainingCV.mat');
assert(isfile(p),'Run R2_SelectHyperparameters first: %s',p);
s = load(p,'alpha_selected','kappa_selected');
alpha=s.alpha_selected; kappa=s.kappa_selected;
assert(isfinite(alpha)&&alpha>0&&isfinite(kappa)&&kappa>0);
end
