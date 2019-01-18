clear all; close all; clc;

load('M');

[m,n] = size(M);
lambda = 1/sqrt(max(size(M)));
beta = 0.01;

[L1,S1] = lq_lq_l2_admm(M, lambda, 1, 1, zeros(m,n), zeros(m,n), beta);
fg1 = M-L1;


L2 = lq_lq_l2_admm(M, lambda, 1, 0.5, L1, S1, beta);
fg2 = M-L2;
 
figure(2);
set(gcf,'outerposition',get(0,'screensize'));
for k =1:size(M,2)
   subplot(321); imshow(reshape(M(1:end,k),[130,160]),[]);title('Convex (q1=q2=1)');
   subplot(323); imshow(reshape(L1(1:end,k),[130,160]),[]);
   subplot(325); imshow(reshape(fg1(1:end,k),[130,160]),[]);
   subplot(322); imshow(reshape(M(1:end,k),[130,160]),[]);title('Nonconvex (q1=1,q2=0.5)');
   subplot(324); imshow(reshape(L2(1:end,k),[130,160]),[]);
   subplot(326); imshow(reshape(fg2(1:end,k),[130,160]),[]);
   pause(.1);
end
