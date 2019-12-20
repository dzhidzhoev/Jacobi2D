#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define  Max(a,b) ((a)>(b)?(a):(b))
#define  Min(a,b) ((a)<(b)?(a):(b))

// #define  N   (512 + 2)
int N;
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double eps;
int it;
double **A, **B;
int Nprocs = 1;
int LeftCol, RightCol; // обрабатываем [LeftCol, RightCol) строки
int TagLeft = 10, TagRight = 11, Rank;
int BASE; // смещение для пересчета теоретического номера 
		// строки в индекс в матрице (чтобы хранить в памяти каждого процесса лишь
		// нужную часть матрицы)

void relax();
void resid();
void init();
void verify(); 

int main(int an, char **as)
{
	N = atoi(as[2]) + 2;

	if (MPI_Init(&an, &as) != MPI_SUCCESS) {
		fprintf(stderr, "failed to init MPI\n");
		exit(1);
	}
	if (MPI_Comm_size(MPI_COMM_WORLD, &Nprocs) != MPI_SUCCESS || MPI_Comm_rank(MPI_COMM_WORLD, &Rank) != MPI_SUCCESS) {
		fprintf(stderr, "failed to get communicator size or rank\n");
		exit(1);
	}

	int M = N / Nprocs;
	LeftCol = M * Rank;
	BASE = LeftCol - 2;
	RightCol = LeftCol + M;
	if (Rank == Nprocs - 1) {
		RightCol = N;
	}
	A = malloc(sizeof(*A) * (M + 5));
	B = malloc(sizeof(*B) * (M + 5));
	for (int i = 0; i < M + 5; i++) {
		A[i] = malloc(sizeof(**A) * N);
		B[i] = malloc(sizeof(**B) * N);
	}

	double time = MPI_Wtime();
	double maxtime;

	init();
	MPI_Status s;
	int LeftRank = (Rank > 0 ? Rank - 1 : MPI_PROC_NULL);
	int RightRank = (Rank < Nprocs - 1 ? Rank + 1 : MPI_PROC_NULL);
	for(it=1; it<=itmax; it++)
	{
		double geps;
		eps = 0.;
		MPI_Sendrecv(&A[LeftCol - BASE][0], N, MPI_DOUBLE_PRECISION, LeftRank, TagLeft, &A[RightCol - BASE][0], N, MPI_DOUBLE_PRECISION,
			RightRank, TagLeft, MPI_COMM_WORLD, &s);
		MPI_Sendrecv(&A[RightCol - 1 - BASE][0], N, MPI_DOUBLE_PRECISION, RightRank, TagRight, &A[LeftCol - 1 - BASE][0], N, MPI_DOUBLE_PRECISION,
			LeftRank, TagRight, MPI_COMM_WORLD, &s);
		relax();
		resid();
		if (MPI_Allreduce(&eps, &geps, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) {
			fprintf(stderr, "Failed to reduce it %d\n", it);
			exit(1);
		}
		if (!Rank) {
			printf( "it=%4i   eps=%f\n", it,geps);
		}
		if (geps < maxeps) break;
	}
	MPI_Sendrecv(&A[LeftCol - BASE][0], N, MPI_DOUBLE_PRECISION, LeftRank, TagLeft, &A[RightCol - BASE][0], N, MPI_DOUBLE_PRECISION,
		RightRank, TagLeft, MPI_COMM_WORLD, &s);
	MPI_Sendrecv(&A[RightCol - 1 - BASE][0], N, MPI_DOUBLE_PRECISION, RightRank, TagRight, &A[LeftCol - 1 - BASE][0], N, MPI_DOUBLE_PRECISION,
		LeftRank, TagRight, MPI_COMM_WORLD, &s);
	verify();
	time = MPI_Wtime() - time;
	MPI_Allreduce(&time, &maxtime, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD);
	
	if (!Rank) {
		printf("execution time %fs\n", maxtime);
	}

	MPI_Finalize();

	return 0;
}

void init()
{ 
	for(i=LeftCol; i<RightCol; i++)
	for(j=0; j<=N-1; j++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1) A[i - BASE][j]= 0.;
		else A[i - BASE][j]= ( 1. + i + j ) ;
	}
} 

void relax()
{
	for(i=Max(LeftCol, 1); i<=Min(RightCol - 1, N-2); i++)
	for(j=1; j<=N-2; j++)
	{
		B[i - BASE][j]=(A[i-1 - BASE][j]+A[i+1 - BASE][j]+A[i - BASE][j-1]+A[i - BASE][j+1])/4.;
	}
}

void resid()
{ 
	double local_eps;
	for(i=Max(LeftCol, 1); i<=Min(RightCol - 1, N-2); i++) {
		local_eps = eps;
		for(j=1; j<=N-2; j++)
		{
			double e;
			e = fabs(A[i - BASE][j] - B[i - BASE][j]);         
			A[i - BASE][j] = B[i - BASE][j]; 
			local_eps = Max(eps,e);
		}

		eps = Max(eps, local_eps);
	}
}

void verify()
{
	double s, gs;
	s=0.;
	for(i=LeftCol; i<=RightCol-1; i++)
	for(j=0; j<=N-1; j++)
	{
		s=s+A[i - BASE][j]*(i+1)*(j+1)/(N*N);
	}
	MPI_Allreduce(&s, &gs, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD);
	if (!Rank) {
		printf("  S = %f\n",gs);
	}
}
