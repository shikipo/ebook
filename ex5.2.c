#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Sequential reduction: root process collects and sums all values
void reduce_sequential(int* local_array, int local_size, int* result, int rank, int num_procs) {
    if (rank == 0) {
        int* gathered_data = (int*)malloc(local_size * num_procs * sizeof(int));
        MPI_Gather(local_array, local_size, MPI_INT, gathered_data, local_size, MPI_INT, 0, MPI_COMM_WORLD);

        *result = 0;
        for (int i = 0; i < local_size * num_procs; i++) {
            *result += gathered_data[i];
        }

        free(gathered_data);
    } else {
        MPI_Gather(local_array, local_size, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

// Tree-based reduction
void reduce_tree(int* local_array, int local_size, int* result, int rank, int num_procs) {
    int local_sum = 0;

    // Compute local sum
    for (int i = 0; i < local_size; i++) {
        local_sum += local_array[i];
    }

    int step = 1;
    while (step < num_procs) {
        if (rank % (2 * step) == 0) {
            // Receive from a child process
            if (rank + step < num_procs) {
                int received_sum;
                MPI_Recv(&received_sum, 1, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_sum += received_sum;
            }
        } else {
            // Send to a parent process
            int parent = rank - step;
            MPI_Send(&local_sum, 1, MPI_INT, parent, 0, MPI_COMM_WORLD);
            break; // Exit loop after sending
        }
        step *= 2;
    }

    if (rank == 0) {
        *result = local_sum;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int global_size = 8; // Total number of elements
    int local_size = global_size / num_procs;
    int* global_array = NULL;
    int* local_array = (int*)malloc(local_size * sizeof(int));
    int sequential_result = 0, tree_result = 0;

    if (rank == 0) {
        global_array = (int*)malloc(global_size * sizeof(int));

        // Fill the array with random numbers
        srand(time(NULL));
        printf("Global array:\n");
        for (int i = 0; i < global_size; i++) {
            global_array[i] = rand() % 10 + 1; // Random numbers between 1 and 10
            printf("%d ", global_array[i]);
        }
        printf("\n");
    }

    // Scatter the global array
    MPI_Scatter(global_array, local_size, MPI_INT, local_array, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform sequential reduction
    reduce_sequential(local_array, local_size, &sequential_result, rank, num_procs);

    // Perform tree-based reduction
    reduce_tree(local_array, local_size, &tree_result, rank, num_procs);

    // Validate results in the root process
    if (rank == 0) {
        printf("Sequential reduction result: %d\n", sequential_result);
        printf("Tree-based reduction result: %d\n", tree_result);

        free(global_array);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}
