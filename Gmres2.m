function [x, error, iter, flag] = Gmres2( A, x, b, M, restrt, max_it, tol )

% [x, error, iter, flag] = gmres( A, x, b, M, restrt, max_it, tol )
%
% gmres.m solves the linear system Ax=b
% using the Generalized Minimal residual ( GMRESm ) method with restarts .
%
% input   A        nxn  matrix
%         x        initial guess vector
%         b        right hand side vector
%         M        REAL preconditioner matrix
%         restrt   iterations between restarts
%         max_it   maximum number of iterations
%         tol      error tolerance
%
% output  x        solution vector
%         error    error norm
%         iter     iterations performed
%         flag     0 = solution found 
%                  1 = no convergence 

% initialization
   iter = 0;                                         
   flag = 0;

   bnrm2 = norm( b );
   if  ( bnrm2 == 0.0 ), bnrm2 = 1.0; end

   r = M \ ( b-A*x );
   error = norm( r ) / bnrm2;
   if ( error < tol ) return, end

   % initialize workspace
   [n,n] = size(A);                                  
   m = restrt;
   V = zeros(n,m+1);
   H = zeros(m+1,m);
   cs = zeros(m,1);
   sn = zeros(m,1);
   e1    = zeros(n,1);
   e1(1) = 1.0;

   % begin iteration
   for iter = 1:max_it,                              
    
      r = M \ ( b-A*x );
      V(:,1) = r / norm( r );
      s = norm( r )*e1;
    
      % construct orthonormal
      % basis using Gram-Schmidt
      for i = 1:m,                                   
        w = M \ (A*V(:,i));                          
        
        for k = 1:i,
             H(k,i)= w'*V(:,k);
             w = w - H(k,i)*V(:,k);
        end
        
        H(i+1,i) = norm( w );
        V(:,i+1) = w / H(i+1,i);
            
        % apply Givens rotation
        for k = 1:i-1,                              
            temp     =  cs(k)*H(k,i) + sn(k)*H(k+1,i);
            H(k+1,i) = -sn(k)*H(k,i) + cs(k)*H(k+1,i);
            H(k,i)   = temp;
        end
        
        % form i-th rotation matrix
        [cs(i),sn(i)] = rotmat( H(i,i), H(i+1,i) ); 
        
        % approximate residual norm
        temp   = cs(i)*s(i);                        
        s(i+1) = -sn(i)*s(i);
        s(i)   = temp;
        H(i,i) = cs(i)*H(i,i) + sn(i)*H(i+1,i);
        H(i+1,i) = 0.0;
        error  = abs(s(i+1)) / bnrm2;
                
        % update approximation
        % and exit
        if ( error <= tol ),                         
            y = H(1:i,1:i) \ s(1:i);                 
            x = x + V(:,1:i)*y;
            break;
        end
        
      end

      if ( error <= tol ), break, end
     
      y = H(1:m,1:m) \ s(1:m);
      % update approximation
      x = x + V(:,1:m)*y;   
      % compute residual
      r = M \ ( b-A*x );                             
      s(i+1) = norm(r);
      % check convergence
      error = s(i+1) / bnrm2;                        
      
      if ( error <= tol ), break, end;
   
   end

   % solution found
   if ( error > tol ) flag = 1; end;                 





