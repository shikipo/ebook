#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void prefix_mpi(int* local_in, int local_size, int** local_outp, int rank, int num_procs) {
    int* local_out = (int*)malloc(local_size * sizeof(int));
    int local_sum = 0;

    // 1st Step: Compute local sums
    for (int i = 0; i < local_size; i++) {
        local_sum += local_in[i];
    }

    int prefix_sum = 0;

    // 2nd Step: Compute prefix sum of local sums using MPI_Scan
    MPI_Scan(&local_sum, &prefix_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Offset correction for local array
    prefix_sum -= local_sum;

    // 3rd Step: Compute prefix sums locally
    local_out[0] = prefix_sum;
    for (int i = 1; i < local_size; i++) {
        local_out[i] = local_out[i - 1] + local_in[i - 1];
    }

    // Return the local output array
    *local_outp = local_out;
}

void prefix_sequential(int* array, int length, int* result) {
    result[0] = 0;
    for (int i = 1; i < length; i++) {
        result[i] = result[i - 1] + array[i - 1];
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int in_length = 8; // Length of the array
    int* global_array = NULL;
    int* global_result_parallel = NULL;
    int* global_result_sequential = NULL;

    int local_size = in_length / num_procs; // Size of local data for each process
    int* local_array = (int*)malloc(local_size * sizeof(int));
    int* local_result = NULL;

    if (rank == 0) {
        // Generate random input array
        global_array = (int*)malloc(in_length * sizeof(int));
        global_result_parallel = (int*)malloc(in_length * sizeof(int));
        global_result_sequential = (int*)malloc(in_length * sizeof(int));

        srand(time(NULL));
        printf("Input array:\n");
        for (int i = 0; i < in_length; i++) {
            global_array[i] = rand() % 11; // Random numbers between 0 and 10
            printf("%d ", global_array[i]);
        }
        printf("\n");

        // Compute sequential prefix sum for validation
        prefix_sequential(global_array, in_length, global_result_sequential);
        printf("Prefix sum computed sequentially:\n");
        for (int i = 0; i < in_length; i++) {
            printf("%d ", global_result_sequential[i]);
        }
        printf("\n");
    }

    // Scatter global array to all processes
    MPI_Scatter(global_array, local_size, MPI_INT, local_array, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute prefix sum in parallel
    prefix_mpi(local_array, local_size, &local_result, rank, num_procs);

    // Gather results from all processes
    MPI_Gather(local_result, local_size, MPI_INT, global_result_parallel, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Validate and print results
    if (rank == 0) {
        printf("Prefix sum computed in parallel:\n");
        for (int i = 0; i < in_length; i++) {
            printf("%d ", global_result_parallel[i]);
        }
        printf("\n");

        free(global_array);
        free(global_result_parallel);
        free(global_result_sequential);
    }

    free(local_array);
    free(local_result);

    MPI_Finalize();
    return 0;
}