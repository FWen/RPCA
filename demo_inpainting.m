clear all; clc; % close all;

m = 256;
n = m;
X = imresize(double(imread('boat.png')),[m, n]);

[S,V,D] = svd(X);
v = diag(V);

%---strictly low-rank-----
v(round(0.15*m):end) = 0;
X = S*diag(v)*D';

%---salt-and-pepper noise-----
c_ratio = 0.2; % corruption ratio
J = randperm(m*n); 
J = J(1:round(c_ratio*m*n));    
noise = randn(1,round(c_ratio*m*n));
noise(find(noise>=0))=255;
noise(find(noise<0))=0;
y = X(:);
y(J) = noise;

Strue = reshape(y, m, n) - X;


% ----- small entry-wise noise ----------
% small_noise = 2*randn(size(y));
% y = y + small_noise;


figure(1);subplot(1,3,1); imshow(uint8(reshape(y, m, n)));
title(sprintf('Corrupted\n RelErr=%.3f, PSNR=%.2f dB',norm(y - X(:))/norm(X(:)),psnr(X(:),y)));


% --- soft thresholding --------------------
lamdas = logspace(-3, 0, 30);
parfor k = 1:length(lamdas); 
    [L, S, out] = lq_lq_l2_admm(reshape(y, m, n), lamdas(k), 1, 1, zeros(m,n), zeros(m,n), X, Strue);
    relerr1(k)  = norm(L-X,'fro')/norm(X,'fro');
    X_l1(:,:,k) = L;
    S_l1(:,:,k) = S;
end
% figure(2);semilogy(lamdas,relerr1,'r-*');set(gca,'xscale','log');grid;


[RelErr1, mi] = min(relerr1);
X_L1 = X_l1(:,:,mi); 
S_L1 = S_l1(:,:,mi); 

figure(1);subplot(1,3,2);
imshow(uint8(X_L1));
title(sprintf('Soft\n RelErr=%.3f, PSNR=%.2f dB',RelErr1,psnr(X(:),X_L1(:))));


% ----- L0 thresholding ----------

parfor k = 1:length(lamdas)           
    [L, ~, ~] = lq_lq_l2_admm(reshape(y, m, n), lamdas(k), 0, 0, X_L1, S_L1, X, Strue);
    relerr(k) = norm(L-X,'fro')/norm(X,'fro');
    X_L0(:,:,k) = L;
end

% figure(3);semilogy(lamdas,relerr,'r-*');set(gca,'xscale','log');grid;


[RelErr, mi] = min(relerr); 
PSNR = psnr(X, X_L0(:,:,mi));

figure(1);subplot(1,3,3);
imshow(uint8(X_L0(:,:,mi)));
title(sprintf('Hard\n RelErr=%.2e, PSNR=%.2f dB',RelErr,PSNR));
