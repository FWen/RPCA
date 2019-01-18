function [L,S,out] = lq_lq_l2_admm(M,lamda,q1,q2,L0,S0,beta);
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
MAX_ITER = 20; 
ABSTOL = 1e-6;

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

% out.el = []; 
% out.es = [];

for iter = 1:MAX_ITER
       
    Lm1 = L; Sm1 = S;
    Zm1 = Z; Vm1 = V;
	
%     beta = beta*1.1;
    rho1 = 5*beta;
    rho2 = rho1;

    % Z-update 
    [E,A,D] = svd(L+W/rho1,'econ');
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
    
    %Check for convergence
    if (norm(L-Lm1,'fro')< sqrt(m*n)*ABSTOL) & (norm(S-Sm1,'fro')< sqrt(m*n)*ABSTOL)
        break;           
    end

end

end
