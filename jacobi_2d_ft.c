#include <mpi.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const int ALIVE_COUNT_TAG = 100;
const int ALIVE_RANKS_TAG = 101;
const int LEADER_TAG = 102;
const int LEADER_ALIVE_COUNT_TAG = 103;
const int LEADER_ALIVE_RANKS_TAG = 104;

void
ignore_mpi_error(MPI_Comm *comm, int *code, ...)
{
}

static MPI_Errhandler MPI_IGNORE_ERROR;

static MPI_Comm current_comm = MPI_COMM_WORLD;

static int leader_rank;
static int alive_count;
static int *alive_ranks;
static int current_size;
static int current_rank;

static void found_error();
static void restart_checkpoint();

void finalize_success()
{
    MPI_Finalize();
    exit(0);
}

static void 
setup_handlers(MPI_Comm comm)
{
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
}

static int
MPI_Elect_leader(MPI_Comm comm, int *alive_count_output, int **alive_ranks_output)
{
    int comm_size = current_size, rank = current_rank;

    MPI_Barrier(comm);

    // устанавливаем обработчик, игнорирующий ошибки
    MPI_Errhandler old_errhandler = NULL;
    if (MPI_Comm_get_errhandler(comm, &old_errhandler) != MPI_SUCCESS)
    {
		fprintf(stderr, "failed to get error handler\n");
		exit(1);
    } 
    if (MPI_Comm_set_errhandler(comm, MPI_IGNORE_ERROR) != MPI_SUCCESS)
    {
        fprintf(stderr , "failed to set ignoring error handler\n");
        exit(1);
    }


    int alive_count = 0;
    int *alive_ranks = NULL;
    int prev = rank - 1;
    //  пытаемся получить список живых процессов с меньшим рангом
    while (prev >= 0)
    {
        MPI_Status status;
        printf("%d trying to receive count from %d\n", current_rank, prev);
        if (MPI_Recv(&alive_count, 1, MPI_INT, prev, ALIVE_COUNT_TAG, comm, &status) 
                == MPI_SUCCESS)
        {
            printf("trying to receive list from %d ac %d\n", prev, alive_count);
            alive_ranks = realloc(alive_ranks, sizeof(*alive_ranks) * alive_count);
            if (MPI_Recv(alive_ranks, alive_count, MPI_INT, prev, ALIVE_RANKS_TAG, comm, &status)
                    == MPI_SUCCESS)
            {
                printf("rcvd\n");
                break;
            }
            else
            {
                fprintf(stderr, "unable to receive list of alive processes\n");
            }
        }
        --prev;
    }
    printf("prevs found %d\n", current_rank);
    // добавляем текущий процесс в список живых
    ++alive_count;
    alive_ranks = realloc(alive_ranks, sizeof(*alive_ranks) * alive_count);
    alive_ranks[alive_count - 1] = rank;
    // пытаемся передать список живых процессов процессу с большим рангом
    int next = rank + 1;
    bool is_leader = true;
    while (next < comm_size)
    {
        printf("send count to %d\n", next);
        if (MPI_Send(&alive_count, 1, MPI_INT, next, ALIVE_COUNT_TAG, comm) == MPI_SUCCESS)
        {
            printf("send list to %d ac %d\n", next, alive_count);
            if (MPI_Send(alive_ranks, alive_count, MPI_INT, next, ALIVE_RANKS_TAG, comm) 
                    == MPI_SUCCESS)
            {
                is_leader = false;
                break;
            }
            else
            {
                fprintf(stderr, "unable to send list of alive processes\n");
            }
        }
        ++next;
    }

    // лидер выбран, фиксируем список живых процессов
   
    // восстанавливаем обработчик ошибок
    MPI_Comm_set_errhandler(comm, old_errhandler);

    // завершаем выборы лидера
    MPI_Barrier(comm);
    printf("Elections completed\n");

    int leader_rank = -1;
    if (is_leader)
    {
        printf("i am leadr %d\n", rank);
        printf("alive is %d\n", alive_count);
        leader_rank = rank;
        *alive_count_output = alive_count;
        // рассылаем всем, кроме себя, информацию о лидере
        for (int i = 0; i < alive_count - 1; ++i)
        {
            printf("%d is alive (msg from leader)\n", alive_ranks[i]);
            if (MPI_Send(&rank, 1, MPI_INT, alive_ranks[i], LEADER_TAG, comm)
                        != MPI_SUCCESS
                    || MPI_Send(&alive_count, 1, MPI_INT, 
                        alive_ranks[i], LEADER_ALIVE_COUNT_TAG, comm)
                        != MPI_SUCCESS
                    || MPI_Send(alive_ranks, alive_count, MPI_INT, 
                        alive_ranks[i], LEADER_ALIVE_RANKS_TAG, comm)
                        != MPI_SUCCESS)
            {
                fprintf(stderr, "unable to send leader tag to %d\n", alive_ranks[i]);
                found_error();
            }
        }
    }
    else
    {
        MPI_Status status;
        printf("i am not leadr %d\n", rank);
        if (MPI_Recv(&leader_rank, 1, MPI_INT, MPI_ANY_SOURCE, LEADER_TAG, comm, &status)
                    != MPI_SUCCESS
                || MPI_Recv(alive_count_output, 1, MPI_INT,
                        MPI_ANY_SOURCE, LEADER_ALIVE_COUNT_TAG, comm, &status)
                    != MPI_SUCCESS)
        {
            fprintf(stderr, "unable to receive leader tag at %d\n", rank);
            found_error();
        }
        printf("alive is %d (from %d)\n", *alive_count_output, current_rank);
        alive_ranks = realloc(alive_ranks, sizeof(*alive_ranks) * *alive_count_output);
        if (MPI_Recv(alive_ranks, *alive_count_output, MPI_INT, MPI_ANY_SOURCE, 
                    LEADER_ALIVE_RANKS_TAG, comm, &status) != MPI_SUCCESS)
        {
            fprintf(stderr, "unable to receive leader tag at %d\n", rank);
            found_error();
        }
        printf("rcvd\n");
    }
    *alive_ranks_output = alive_ranks;

    printf("Function work done %d\n", current_rank);

    return leader_rank;
}

void
found_error()
{
    MPIX_Comm_revoke(current_comm);

    MPI_Comm next_comm = MPI_COMM_NULL;
    MPIX_Comm_shrink(current_comm, &next_comm);
    setup_handlers(next_comm);
    if (MPI_Comm_size(next_comm, &current_size) != MPI_SUCCESS || 
            MPI_Comm_rank(next_comm, &current_rank) != MPI_SUCCESS)
    {
		fprintf(stderr, "failed to get communicator size or rank\n");
		exit(1);
    }
    current_comm = next_comm;
    
    leader_rank = MPI_Elect_leader(current_comm, &alive_count, &alive_ranks);
    restart_checkpoint();
}

#define  Max(a,b) ((a)>(b)?(a):(b))
#define  Min(a,b) ((a)<(b)?(a):(b))

// #define  N   (512 + 2)
int N = 512 + 2;
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double eps;
int it = -1;
double **A = NULL, **B = NULL; // матрицы
int LeftCol, RightCol; // обрабатываем [LeftCol, RightCol) строки
int TagLeft = 10, TagRight = 11;
int BASE; // смещение для пересчета теоретического номера 
		// строки в индекс в матрице (чтобы хранить в памяти каждого процесса лишь
		// нужную часть матрицы)
MPI_Status s;

void
find_current_task()
{
    int Nprocs = current_size;
    int Rank = current_rank;
    int M = N / Nprocs;
    LeftCol = M * Rank;
    BASE = LeftCol - 2;
    RightCol = LeftCol + M;
    if (Rank == Nprocs - 1)
    {
		// на случай, если количество строк матрицы не делится на количество
		// решающих задачу процессов
		RightCol = N;
    }
    free(A);
    free(B);

	A = malloc(sizeof(*A) * (M + 5));
	B = malloc(sizeof(*B) * (M + 5));
	for (int i = 0; i < M + 5; i++) {
		A[i] = malloc(sizeof(**A) * N);
		B[i] = malloc(sizeof(**B) * N);
	}

    printf("%d task is %d %d\n", current_rank, LeftCol, RightCol);
}

void 
save_checkpoint(double **mat)
{
    for (int i = LeftCol; i < RightCol; ++i)
    {
        char buf[512];
        sprintf(buf, "%d_%d.txt", it, i);
        FILE *f = fopen(buf, "w");
        if (!f)
        {
            fprintf(stderr, "failed to save backup %d!\n", i);
            found_error();
        }

        fprintf(f, "%f\n", eps);
        for(int j=0; j<=N-1; j++)
        {
            fprintf(f, "%f\n", mat[i - BASE][j]);
        }

        if (fclose(f))
        {
            fprintf(stderr, "failed to save backup %d!\n", i);
            found_error();
        }
    }
}

void
load_checkpoint(int it, double **mat)
{
    it = -1;
    for (int i = LeftCol; i < RightCol; ++i)
    {
        char buf[512];
        sprintf(buf, "%d_%d.txt", it, i);
        FILE *f = fopen(buf, "r");
        if (!f)
        {
            fprintf(stderr, "failed to load backup %d!\n", i);
            found_error();
        }

        fscanf(f, "%lf\n", &eps);
        for(int j=0; j<=N-1; j++)
        {
            fscanf(f, "%lf\n", &mat[i - BASE][j]);
        }

        if (fclose(f))
        {
            fprintf(stderr, "failed to load backup %d!\n", i);
            found_error();
        }
    }
}

bool 
has_checkpoint(int it)
{
    for (int i = LeftCol; i < RightCol; ++i)
    {
        char buf[512];
        sprintf(buf, "%d_%d.txt", it, i);
        FILE *f = fopen(buf, "r");
        if (!f)
        {
            return false;
        }
        fclose(f);   
    }
    return true;
}

void
find_and_load_checkpoint()
{
    // находим номер последней сохраненной итерации для столбцов матрицы
    // за которые отвечает данный процесс (при помощи бин. поиска)
    //
    // синхронизируем номера точек сохранения, и загружаем последнюю
    int lower = -1, upper = it + 1;
    while (lower + 1 < upper)
    {
        int mid = (upper + lower) / 2;

        if (has_checkpoint(mid))
        {
            lower = mid;
        }
        else
        {
            upper = mid;
        }
    }
    if (lower >= upper)
    {
        it = -1;
    }
    else
    {
        it = lower;
    }
    int global_it;
    if (MPI_Allreduce(&it, &global_it, 1, MPI_INT, MPI_MIN, current_comm) != MPI_SUCCESS)
    {
        fprintf(stderr, "unable to find out last saved iteration\n");
        found_error();
    }
    it = global_it;

    if (it >= 0)
    {
        load_checkpoint(it, A);
    }
}

void init();
void relax();
void resid();
void verify();

static void
restart_checkpoint()
{
    find_current_task();
    find_and_load_checkpoint();

    if (it < 0)
    {
        printf("before init\n");
        init();
        printf("after init\n");
        save_checkpoint(A);
    }
	int LeftRank = (current_rank > 0 ? current_rank - 1 : MPI_PROC_NULL);
	int RightRank = (current_rank < current_size - 1 ? current_rank + 1 : MPI_PROC_NULL);
    for (; it <= itmax;)
    {
        double geps;
        eps = 0.;
		if (MPI_Sendrecv(&A[LeftCol - BASE][0], N, MPI_DOUBLE_PRECISION, LeftRank, TagLeft, &A[RightCol - BASE][0], N, MPI_DOUBLE_PRECISION,
			RightRank, TagLeft, MPI_COMM_WORLD, &s) != MPI_SUCCESS)
        {
            fprintf(stderr, "unable to exchange with left\n");
            found_error();
        }
		if (MPI_Sendrecv(&A[RightCol - 1 - BASE][0], N, MPI_DOUBLE_PRECISION, RightRank, TagRight, &A[LeftCol - 1 - BASE][0], N, MPI_DOUBLE_PRECISION,
			LeftRank, TagRight, MPI_COMM_WORLD, &s) != MPI_SUCCESS)
        {
            fprintf(stderr, "unable to exchange with right\n");
            found_error();
        }
		relax();
		resid();
		if (MPI_Allreduce(&eps, &geps, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) {
			fprintf(stderr, "Failed to reduce it %d\n", it);
            found_error();
		}
        int Rank = current_rank;
		if (!Rank) {
			printf( "it=%4i   eps=%f\n", it,geps);
		}
		if (geps < maxeps) break;

        // точка сохранения
        ++it;
        save_checkpoint(A);

        // случайно падаем
        if (rand() % 1000 < 50)
        {
            printf("suicide %d!!!\n", current_rank);
            exit(1);
        }
    }

	if (MPI_Sendrecv(&A[LeftCol - BASE][0], N, MPI_DOUBLE_PRECISION, LeftRank, TagLeft, &A[RightCol - BASE][0], N, MPI_DOUBLE_PRECISION,
		RightRank, TagLeft, MPI_COMM_WORLD, &s) != MPI_SUCCESS)
    {
        fprintf(stderr, "unable to exchange with left\n");
        found_error();
    }
	if (MPI_Sendrecv(&A[RightCol - 1 - BASE][0], N, MPI_DOUBLE_PRECISION, RightRank, TagRight, &A[LeftCol - 1 - BASE][0], N, MPI_DOUBLE_PRECISION,
		LeftRank, TagRight, MPI_COMM_WORLD, &s) != MPI_SUCCESS)
    {
        fprintf(stderr, "unable to exchange with right\n");
        found_error();
    }
	verify();

    finalize_success();
}

void init()
{ 
	for(i=LeftCol; i<RightCol; i++)
	for(j=0; j<=N-1; j++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1) A[i - BASE][j]= 0.;
		else A[i - BASE][j]= ( 1. + i + j ) ;
	}
    it = 0;
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
	if (MPI_Allreduce(&s, &gs, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD)
            != MPI_SUCCESS)
    {
        fprintf(stderr, "failed to exchange S\n");
        found_error();
    }
    int Rank = current_rank;
	if (!Rank) {
		printf("  S = %f\n",gs);
	}
}

int 
main(int an, char **as)
{
    if (MPI_Init(&an, &as) != MPI_SUCCESS) 
    {
		fprintf(stderr, "failed to init MPI\n");
		exit(1);
	}

    if (MPI_SUCCESS != MPI_Comm_create_errhandler(ignore_mpi_error, &MPI_IGNORE_ERROR))
    {
		fprintf(stderr, "failed create ignoring error handler\n");
		exit(1);
    }

    MPI_Comm_dup(current_comm, &current_comm);
    setup_handlers(current_comm);
    
    if (MPI_Comm_rank(current_comm, &current_rank) != MPI_SUCCESS)
    {
		fprintf(stderr, "failed to get current process rank\n");
		exit(1);
    }    
    srand(time(0) + 1000 * current_rank);

    found_error();
        
	MPI_Finalize();

    return 0;
}
