function [B,base] = R2_SparseLandmarkBasis(coeff,meanV,small_size,img_size,pts)
% Exact same resize and landmark interpolation as dense reconstruction.
% Offline precomputation of all 3 displacement components (not just x).
n = size(pts,1); k = size(coeff,2); N=prod(small_size);
B=zeros(3*n,k); base=zeros(3*n,1);
for c=1:3
    idx=(c-1)*N+(1:N); rows=(c-1)*n+(1:n);
    v=imresize3(reshape(meanV(idx),small_size),img_size,'linear');
    base(rows)=interp3(double(v),pts(:,1),pts(:,2),pts(:,3),'linear',0);
    for j=1:k
        v=imresize3(reshape(coeff(idx,j),small_size),img_size,'linear');
        B(rows,j)=interp3(double(v),pts(:,1),pts(:,2),pts(:,3),'linear',0);
    end
end
end
