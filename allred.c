#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <mpi-ext.h>

#define  Max(a,b) ((a)>(b)?(a):(b))

const int DIMS[] = {4, 4};
const int NDIMS = sizeof(DIMS) / sizeof(*DIMS);
const int PERIOD[] = {0, 0};

void
MPI_Allreduce_dim(MPI_Comm comm, int rank, const int coords[NDIMS], double *val, double *max,
        int next_coords[NDIMS], int prev_coords[NDIMS], int dim)
{
    const int TAG = 100;
    int middle = DIMS[dim] / 2;
    int next_rank;
    MPI_Status status;
    MPI_Cart_rank(comm, next_coords, &next_rank);
    if (coords[dim] > 0 && coords[dim] < DIMS[dim] - 1)
    {
        int prev_rank;
        MPI_Status status;
        MPI_Cart_rank(comm, prev_coords, &prev_rank);
        double buf;
        MPI_Recv(&buf, 1, MPI_DOUBLE_PRECISION, prev_rank, TAG, comm, &status);
//        printf("recv %d %d %d %d %f\n", coords[0], coords[1], prev_coords[0], prev_coords[1], buf);
        *max = Max(*max, buf);
    }
    else
    {
    }
    if (coords[dim] == middle || (DIMS[dim] % 2 == 0 && coords[dim] == middle - 1))
    {
        int prev_rank;
        MPI_Status status;
        MPI_Cart_rank(comm, next_coords, &prev_rank);

        if (dim == 0)
        {
            int i = coords[0];
            int j = coords[1];
            int middle_j = DIMS[1] / 2;
            int next_j = j + (j < middle_j ? 1 : -1);
            int prev_j = j + (j < middle_j ? -1 : 1);
            int next_j_coords[NDIMS] = {coords[0], next_j};
            int prev_j_coords[NDIMS] = {coords[0], prev_j};
            MPI_Allreduce_dim(comm, rank, coords, val, max, next_j_coords, prev_j_coords, 1);
        }
        else
        {
            // центр матрицы
            if (coords[dim] != middle)
            {
                int next_rank;
                int next_coord[] = { coords[0], middle };
                MPI_Cart_rank(comm, next_coord, &next_rank);
                MPI_Rsend(max, 1, MPI_DOUBLE_PRECISION, next_rank, TAG, comm);
                double buf;
                MPI_Recv(&buf, 1, MPI_DOUBLE_PRECISION, next_rank, TAG, comm, &status);
//                printf("recv %d %d %d %d %f\n", coords[0], coords[1], next_coords[0], next_coords[1], buf);
                *max = Max(*max, buf);
            }
            else if (DIMS[dim] % 2 == 0)
            {
                int next_rank;
                int next_coord[] = { coords[0], middle - 1 };
                MPI_Cart_rank(comm, next_coord, &next_rank);
                double buf;
                MPI_Recv(&buf, 1, MPI_DOUBLE_PRECISION, next_rank, TAG, comm, &status);
//                printf("recv %d %d %d %d %f\n", coords[0], coords[1], next_coords[0], next_coords[1], buf);
                *max = Max(*max, buf);
                MPI_Rsend(max, 1, MPI_DOUBLE_PRECISION, next_rank, TAG, comm);
            }
            if (DIMS[0] % 2 == 0)
            {
                if (coords[0] != DIMS[0] / 2)
                {
                    int next_rank;
                    int next_coord[] = { DIMS[0] / 2, coords[1] };
                    MPI_Cart_rank(comm, next_coord, &next_rank);
                    MPI_Rsend(max, 1, MPI_DOUBLE_PRECISION, next_rank, TAG, comm);
                    double buf;
                    MPI_Recv(&buf, 1, MPI_DOUBLE_PRECISION, next_rank, TAG, comm, &status);
//                    printf("recv %d %d %d %d %f\n", coords[0], coords[1], next_coords[0], next_coords[1], buf);
                    *max = Max(*max, buf);
                }
                else
                {
                    int next_rank;
                    int next_coord[] = { DIMS[0] / 2 - 1, coords[1] };
                    MPI_Cart_rank(comm, next_coord, &next_rank);
                    double buf;
                    MPI_Recv(&buf, 1, MPI_DOUBLE_PRECISION, next_rank, TAG, comm, &status);
//                    printf("recv %d %d %d %d %f\n", coords[0], coords[1], next_coords[0], next_coords[1], buf);
                    *max = Max(*max, buf);
                    MPI_Rsend(max, 1, MPI_DOUBLE_PRECISION, next_rank, TAG, comm);
                }
            }
        }
        MPI_Cart_rank(comm, prev_coords, &prev_rank);
        MPI_Rsend(max, 1, MPI_DOUBLE_PRECISION, prev_rank, TAG, comm);
    }
    else 
    {
        MPI_Rsend(max, 1, MPI_DOUBLE_PRECISION, next_rank, TAG, comm);
        double buf;
        MPI_Recv(&buf, 1, MPI_DOUBLE_PRECISION, next_rank, TAG, comm, &status);
//        printf("recv %d %d %d %d %f\n", coords[0], coords[1], next_coords[0], next_coords[1], buf);
        *max = Max(*max, buf);
        if (prev_coords[dim] > 0 && prev_coords[dim] < DIMS[dim] - 1)
        {
            int prev_rank;
            MPI_Status status;
            MPI_Cart_rank(comm, prev_coords, &prev_rank);
            MPI_Rsend(max, 1, MPI_DOUBLE_PRECISION, prev_rank, TAG, comm);
        }
    }
}

void 
MPI_Allreduce_task(MPI_Comm comm, int rank, const int coords[NDIMS], double *val, double *max)
{
    int i = coords[0];
    int j = coords[1];
    int middle_i = DIMS[0] / 2;
    int next_i = i + (i < middle_i ? 1 : -1);
    int prev_i = i + (i < middle_i ? -1 : 1);
    int next_i_coords[NDIMS] = {next_i, j};
    int prev_i_coords[NDIMS] = {prev_i, j};
    MPI_Allreduce_dim(comm, rank, coords, val, max, next_i_coords, prev_i_coords, 0);
}

int
main(int argc, char **argv)
{
    srand(time(0));

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "failed to init MPI\n");
        exit(1);
    }

    MPI_Comm matrix_comm;
    if (MPI_Cart_create(MPI_COMM_WORLD, 
                NDIMS, DIMS, PERIOD, true, &matrix_comm) != MPI_SUCCESS) {
        fprintf(stderr, "failed to create topology");
        exit(1);
    }
    
    int cur_rank;
    int cur_coords[NDIMS];
    if (MPI_Comm_rank(matrix_comm, &cur_rank) != MPI_SUCCESS) {
        fprintf(stderr, "failed to get rank");
        exit(1);
    }
    if (MPI_Cart_coords(matrix_comm, cur_rank, NDIMS, cur_coords) != MPI_SUCCESS) {
        fprintf(stderr, "failed to get coords");
        exit(1);
    }
    
    double my = cur_rank;
    double my_max = cur_rank;
    int i = rand() % DIMS[0], j = rand() % DIMS[1];
    if (i == cur_coords[0] && j == cur_coords[1])
    {
        my = 100;
        my_max = 100;
    }
    printf("before val %d %d %f %f\n", cur_coords[0], cur_coords[1], my, my_max);
    MPI_Barrier(matrix_comm); // для отладки
    MPI_Allreduce_task(matrix_comm, cur_rank, cur_coords, &my, &my_max);
    MPI_Barrier(matrix_comm); // для отладки
    printf("after val %d %d %f %f\n", cur_coords[0], cur_coords[1], my, my_max);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
