function [L,S,out] = lq_lq_l2_admm(M,lamda,q1,q2,L0,S0,Ltrue,Strue);
% lq_lq_l2_admm solves
%
%   minimize ||L||_Sq1^Sq1 + lamda*||S||_q2^q2 + \beta/2*||M-L-S||_F^2
%
% Inputs 
%	1=>q1,q2=>0
%	Ltrue, Strue: for debug, for calculation of errors
%   L0,S0: intialization
% Outputs
%	L,S: the recovery
%	out.el, out.es: the error with respect to the true


%Convergence setup
MAX_ITER = 1e3; 
ABSTOL = 1e-9;

[m, n] = size(M);

%Initialize
if nargin<5
	L = zeros(m,n);
    S = zeros(m,n);
else
    L = L0;
    S = S0;
end

isquiet = 0;
if nargin<7
    isquiet = 1;
end

Z = zeros(m,n);
V = zeros(m,n);
W = zeros(m,n);
U = zeros(m,n);

betaT = 5e2;
beta = 5e-5; 

rhoT = 5*betaT;
rho1 = 0.1*beta;

out.el = []; 
out.es = [];

for iter = 1:MAX_ITER

    Lm1 = L; Sm1 = S;
    Zm1 = Z; Vm1 = V;
	
    if beta<betaT % for acceleration of the algorithm
        beta = beta * 1.03;
    end    
    if rho1<rhoT % for acceleration of the algorithm
        rho1 = rho1 * 1.03;
    end
    rho2 = rho1;

    % Z-update 
    [E,A,D] = svd(L+W/rho1);
    a = shrinkage_Lq(diag(A), q1, 1, rho1);
    Z = E*diag(a)*D';
    
    % V-update
    T = S+U/rho2;
    V = reshape(shrinkage_Lq(T(:), q2, lamda, rho2), m, n);
    
    % L- and S-update
    L = (beta*(M-S)+rho1*Z-W)/(beta+rho1);
    S = (beta*(M-L)+rho2*V-U)/(beta+rho2);
    
    % dual variables: W and U
    W = W + rho1 * (L - Z);
    U = U + rho2 * (S - V);
    
    if ~isquiet
        out.el  = [out.el, norm(L-Ltrue,'fro')/norm(Ltrue,'fro')];
        out.es  = [out.es, norm(S-Strue,'fro')/norm(Strue,'fro')];
    end
        
    %Check for convergence
    if (norm(L-Lm1,'fro')< sqrt(m*n)*ABSTOL) & (norm(S-Sm1,'fro')< sqrt(m*n)*ABSTOL)
        break;           
    end

end

end
