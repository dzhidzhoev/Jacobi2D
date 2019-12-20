#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

// #define  N   (512 + 2)
int N;
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double eps;
int it;
double **A, **B;

void relax();
void resid();
void init();
void verify(); 

int main(int an, char **as)
{
#ifndef _OPENMP
	#error No OpenMP.
#endif

	double time = omp_get_wtime();
	omp_set_num_threads(atoi(as[1]));
	N = atoi(as[2]) + 2;
	A = malloc(sizeof(*A) * N);
	B = malloc(sizeof(*B) * N);
	for (int i = 0; i < N; i++) {
		A[i] = malloc(sizeof(*A) * N);
		B[i] = malloc(sizeof(*B) * N);
	}

	init();
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		resid();
		// printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}
	verify();

	printf("execution time %fs\n", omp_get_wtime() - time);

	return 0;
}

void init()
{ 
#pragma omp parallel for shared(A) private(i,j)
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1) A[i][j]= 0.;
		else A[i][j]= ( 1. + i + j ) ;
	}
} 

void relax()
{
#pragma omp parallel for shared(A,B) private(i,j)
	for(i=1; i<=N-2; i++)
	for(j=1; j<=N-2; j++)
	{
		B[i][j]=(A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4.;
	}
}

void resid()
{ 
#pragma omp parallel for private(i,j)
	for(i=1; i<=N-2; i++) {
		double local_eps = eps;
		for(j=1; j<=N-2; j++)
		{
			double e;
			e = fabs(A[i][j] - B[i][j]);         
			A[i][j] = B[i][j]; 
			local_eps = Max(eps,e);
		}

		eps = Max(eps, local_eps);
	}
}

void verify()
{
	double s;
	s=0.;
#pragma omp parallel for reduction(+:s) private(i,j)
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	{
		s=s+A[i][j]*(i+1)*(j+1)/(N*N);
	}
	printf("  S = %f\n",s);
}
