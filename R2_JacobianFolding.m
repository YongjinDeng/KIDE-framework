function fold = R2_JacobianFolding(D, vs, full_size)
% imregdemons components are original-grid voxel displacements (x,y,z).
% imresize3 changes sampling grid, not displacement units.
sz = size(D); ratio = full_size./sz(1:3);
step = vs.*ratio([2 1 3]);
[ux,uy,uz] = gradient(D(:,:,:,1).*vs(1),step(1),step(2),step(3));
[vx,vy,vz] = gradient(D(:,:,:,2).*vs(2),step(1),step(2),step(3));
[wx,wy,wz] = gradient(D(:,:,:,3).*vs(3),step(1),step(2),step(3));
J = (1+ux).*((1+vy).*(1+wz)-vz.*wy) - uy.*(vx.*(1+wz)-vz.*wx) + uz.*(vx.*wy-(1+vy).*wx);
fold = 100*nnz(J<=0)/numel(J);
end
