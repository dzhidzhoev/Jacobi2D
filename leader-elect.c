#include <mpi.h>
#include <signal.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

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

static int leader_rank;
static int alive_count;
static int *alive_ranks;
static int current_rank;

static void found_error();
static void restart_checkpoint();

void finalize_success()
{
    MPI_Finalize();
    exit(0);
}

static int
MPI_Elect_leader(MPI_Comm comm, int *alive_count_output, int **alive_ranks_output)
{
    int comm_size, rank;
    // дожидаемся, пока все процессы в группе начнут выборы лидера
    if (MPI_Barrier(comm) != MPI_SUCCESS)
    {
		fprintf(stderr, "failed to reach barrier\n");
		exit(1);
    }
    
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

    if (MPI_Comm_size(comm, &comm_size) != MPI_SUCCESS || 
            MPI_Comm_rank(comm, &rank) != MPI_SUCCESS)
    {
		fprintf(stderr, "failed to get communicator size or rank\n");
		exit(1);
    }
    printf("handlers has been set!\n");
    int alive_count = 0;
    int *alive_ranks = NULL;
    int prev = rank - 1;
    //  пытаемся получить список живых процессов с меньшим рангом
    while (prev >= 0)
    {
        MPI_Status status;
        printf("trying to receive count from %d\n", prev);
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
    leader_rank = MPI_Elect_leader(MPI_COMM_WORLD, &alive_count, &alive_ranks);
    restart_checkpoint();
}

static void
restart_checkpoint()
{
    if (alive_count == 1)
    {
        printf("I am last alive!!! %d\n", current_rank);
        finalize_success();
    }

    // находим себя в списке живых процессов
    int current_i = -1;
    for (int i = 0; i < alive_count; ++i)
    {
        if (alive_ranks[i] == current_rank)
        {
            current_i = i;
            break;
        }
    }
    if (current_i < 0)
    {
        printf("I am not alive :(\n");
        found_error();
    }

    // гоняем по кругу данные, делаем "полезную" работу
    //  иногда падаем
    int data = 0;
    const int DATA_TAG = 1000;
    if (current_i == 0)
    {
        if (MPI_Send(&data, 1, MPI_INT, alive_ranks[(current_i + 1) % alive_count], 
                    DATA_TAG, MPI_COMM_WORLD) != MPI_SUCCESS)
        {
            fprintf(stderr, "unable to send data\n");
            found_error();
        }
    }
    int j = 0;
    while (j++ < 1000)
    {
        MPI_Status status;
        if (MPI_Recv(&data, 1, MPI_INT, alive_ranks[(current_i - 1 + alive_count) % alive_count],
                    DATA_TAG, MPI_COMM_WORLD, &status) != MPI_SUCCESS)
        {
            fprintf(stderr, "unable to recv data\n");
            found_error();
        }
        printf("process %d recv %d\n", current_rank, data);
        ++data;
        // процесс может упасть
        if (rand() % 100 < 30)
        {
            exit(1);
        }
        if (MPI_Send(&data, 1, MPI_INT, alive_ranks[(current_i + 1) % alive_count], 
                    DATA_TAG, MPI_COMM_WORLD) != MPI_SUCCESS)
        {
            fprintf(stderr, "unable to send data\n");
            found_error();
        }
        sleep(1);
    }

    finalize_success();
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
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    if (MPI_Comm_rank(MPI_COMM_WORLD, &current_rank) != MPI_SUCCESS)
    {
		fprintf(stderr, "failed to get current process rank\n");
		exit(1);
    }    

    found_error();
        
	MPI_Finalize();

    return 0;
}
