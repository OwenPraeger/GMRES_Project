/*
 * GMRES.h
 *
 *  Created on: Apr 23, 2012
 *      Author: opraeger
 */

#ifndef GMRES_H_
#define GMRES_H_

#include <math.h>
#include "armadillo";
#include <Accelerate.h>

using namespace arma;


template < class Mat, class Col>
void
Update(Col &x, int k, Mat &h, Col &s, Col v[])
{
	Col y(s);

	for(int i = k; i>= 0; i--)
	{
		y(i) /= h(i,i);
		for (int j=i-1; j>=0; j--)
		{
			y(j) -= h(j,i)*y(i);
		}
	}
	for (int i=0; i<= k; i++)
	{
		x += v[i]*y(i);
	}
}

template <class Real>
Real
abs(Real x)
{
	return (x > 0 ? x : -x);
}

template <class Mat, class Col, class Real>
bool
GMRES(const Mat &A, Col &x, const Col &b, const Mat &Pre,
		Mat &H, int &restart, int &max_iter, Real &tol)
{

	Real resid;
	int i,j=1,k;
	Col s(restart+1), cs(restart+1), sn(restart+1), w;
	Real normB = norm(solve(Pre, b),2);
	Col r = solve(Pre, b-A*x);
	Real beta = norm(r,2);

	if(normB == 0.0)
		normB = 1.0;

	//if the initial guess vector works to begin with
	if((resid = norm(r,2)/normB)<= tol)
	{
		tol = resid;
		max_iter = 0;
		return true;
	}

	Col *v = new Col[restart+1];
	while (j<= max_iter)
	{
		v[0] = r*(1.0/beta);
		s.fill(0.0);
		s(0) = beta;

		//construct orthonormal basis using Gram-Schmidt
		for(i = 0; i<restart && j<=max_iter; i++, j++)
		{
			w = solve(Pre, A*v[i]);
			for(k=0; k<=i; k++)
			{
				H(k,i) = dot(w,v[k]);
				w -= H(k,i)*v[k];
			}
			H(i+1,i) = norm(w,2);
			v[i+1] = w*(1.0/H(i+1,i));

			//apply Givens rotation
			for(k = 0; k<i; k++)
				ApplyGivens(H(k,i), H(k+1, i), cs(k), sn(k));

			GivensCoefficients(H(i,i),H(i+1,i), cs(i), sn(i));
			ApplyGivens(H(i,i),H(i+1,i), cs(i), sn(i));
			ApplyGivens(s(i), s(i+1), cs(i), sn(i));

			if((resid = abs(s(i+1))/normB) < tol)
			{
				Update(x,i,H,s,v);
				tol = resid;
				max_iter = j;
				delete [] v;
				return true;
			}
		}
		Update(x,i-1, H,s,v);
		r = solve(Pre,b-A*x);
		beta = norm(r,2);
		if ((resid = beta / normB) < tol) {
					tol = resid;
					max_iter = j;
					delete [] v;
					return true;
				}
	}
	tol = resid;
	delete []v;
	return false;
}

template <class Real>
void
GivensCoefficients(Real &dx, Real &dy, Real &cs, Real &sn)
{
	if (dy==0.0)
	{
		cs = 1.0;
		sn = 0.0;
	}
	else if (abs(dy)> abs(dx))
	{
		Real temp = dx/dy;
		sn = 1.0/sqrt(1.0 +temp*temp);
		cs = temp*sn;
	}
	else
	{
		Real temp = dy/dx;
				cs = 1.0/sqrt(1.0 +temp*temp);
				sn = temp*cs;
	}
}

template <class Real>
void
ApplyGivens(Real &dx, Real &dy, Real &cs, Real &sn)
{
	Real temp = cs*dx + sn*dy;
	dy = -sn*dx + cs*dy;
	dx = temp;
}

#endif /* GMRES_H_ */
