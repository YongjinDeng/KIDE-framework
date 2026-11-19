function [noisy, patch, info] = R2_PerturbDepth(clean, vs, seed)
% Shared physical-domain stress test. Depth-map rows=transverse, columns=SI.
assert(all(isfinite(clean(:))), 'Nonfinite depth map; check original cache.');
rng(seed,'twister');
noisy = clean + 2.0*randn(size(clean));
[nr,nc] = size(clean);
h = min(nr,max(1,round(80/vs(1)))); w = min(nc,max(1,round(60/vs(3))));
r0 = floor((nr-h)/2)+1; c0 = floor((nc-w)/2)+1;
patch = false(nr,nc); patch(r0:r0+h-1,c0:c0+w-1) = true;
noisy(patch) = noisy(patch)+80;
info = [h*vs(1), w*vs(3), 100*nnz(patch)/numel(patch), r0, c0, h, w];
end
