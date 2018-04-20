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
c_ratio = 0.25; % corruption ratio
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


figure(1);subplot(1,4,1); imshow(uint8(reshape(y, m, n)));
title(sprintf('Corrupted\n RelErr=%.3f, PSNR=%.2f dB',norm(y - X(:))/norm(X(:)),psnr(X(:),y)));


% --- soft thresholding --------------------
lamdas = logspace(-5, 2, 60);
parfor k = 1:length(lamdas); 
    [L, S, out] = lq_lq_l2_admm(reshape(y, m, n), lamdas(k), 1, 1, zeros(m,n), zeros(m,n), X, Strue);
    relerr1(k)  = norm(L-X,'fro')/norm(X,'fro');
    X_l1(:,:,k) = L;
    S_l1(:,:,k) = S;
end

[RelErr1, mi] = min(relerr1);
X_L1 = X_l1(:,:,mi); 
S_L1 = S_l1(:,:,mi); 

figure(1);subplot(1,4,2);
imshow(uint8(X_L1));
title(sprintf('Soft\n RelErr=%.3f, PSNR=%.2f dB',RelErr1,psnr(X(:),X_L1(:))));


% ----- Lq thresholding ----------
qs = 0:0.2:1;
for l1=1:length(qs)
    for l2=1:length(qs)
        
        t0=tic;
        parfor k = 1:length(lamdas)           
            [L, ~, ~] = lq_lq_l2_admm(reshape(y, m, n), lamdas(k), qs(l1), qs(l2), X_L1, S_L1, X, Strue);
            relerr_admm(k) = norm(L-X,'fro')/norm(X,'fro');
            xx_admm(:,:,k) = L;
        end
        
        [RelErrs_admm(l1,l2), mi] = min(relerr_admm); 
        x_admm(:,l1,l2)  = reshape(xx_admm(:,:,mi), [m*n,1]); 
        PSNR_admm(l1,l2) = psnr(X, xx_admm(:,:,mi));
        
        sprintf('ADMM with q1=%.1f and q2=%.1f completed, elapsed time: %.1f seconds',qs(l1),qs(l2),toc(t0))
    end
end


v0 = min(min(PSNR_admm)); 
v1 = max(max(PSNR_admm));
figure(5);
contourf(qs,qs,PSNR_admm,[v0:10:v1]);    colorbar; xlabel('q_2');ylabel('q_1');
set(gca, 'CLim', [v0, v1]);

figure(1);subplot(1,4,3);
imshow(uint8(reshape(x_admm(:,1,1),[m,n])));
title(sprintf('Hard\n RelErr=%.2e, PSNR=%.2f dB',RelErrs_admm(1,1),PSNR_admm(1,1)));

[w1, e1] = max(PSNR_admm);[~, lo] = max(w1); ko = e1(lo);
figure(1);subplot(1,4,4);
imshow(uint8(reshape(x_admm(:,ko,lo),[m,n])));
title(sprintf('Lq (best, q1=%.1f, q2=%.1f)\n RelErr=%.2e, PSNR=%.2f dB',qs(ko),qs(lo),RelErrs_admm(ko,lo),PSNR_admm(ko,lo)));
