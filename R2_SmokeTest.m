%% Small synthetic tests before accessing the patient caches.
clear; clc;
vs=[1 1 2.5]; clean=zeros(200,100);
[a,mask,info]=R2_PerturbDepth(clean,vs,101);
b=R2_PerturbDepth(clean,vs,101);
assert(isequal(a,b) && nnz(mask)==80*24 && all(info(1:2)==[80 60]));
D=zeros(12,14,10,3,'single'); assert(R2_JacobianFolding(D,vs,[24 28 20])==0);
D(:,:,:,1)=repmat(reshape(single(-3*(0:13)),1,14,1),12,1,10);
assert(R2_JacobianFolding(D,vs,[24 28 20])==100);
rng(21); small=[5 6 4]; full=[10 12 8]; n=prod(small)*3;
C=single(randn(n,3)); mu=single(randn(1,n)); y=single(randn(1,3));
pts=[3 4 2;7 5 6;8.2 3.1 4.4]; [B,base]=R2_SparseLandmarkBasis(C,mu,small,full,pts);
A=reshape(mu+y*C',[small 3]); dense=zeros(size(pts));
for c=1:3
    v=imresize3(A(:,:,:,c),full,'linear');
    dense(:,c)=interp3(double(v),pts(:,1),pts(:,2),pts(:,3),'linear',0);
end
assert(max(abs(dense(:)-(base+B*double(y'))))<1e-5);
% Verify training-only CV quadratic metric against explicit reconstruction.
N=prod(small); e=double(randn(1,3*N)); Q=double(randn(3*N,3)); z=randn(3,1);
G=zeros(3); bb=zeros(3,1); aa=0; direct=0;
for c=1:3
    idx=(c-1)*N+(1:N); M=Q(idx,:); r=e(idx)';
    G=G+vs(c)^2*(M'*M); bb=bb+vs(c)^2*(M'*r); aa=aa+vs(c)^2*(r'*r);
    direct=direct+vs(c)^2*sum((r+M*z).^2);
end
assert(abs((aa+2*z'*bb+z'*G*z)-direct)<1e-8);
fprintf('R2_SMOKE_TEST_PASS\n');
