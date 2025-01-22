#include "timer.hpp"
#include <iomanip>
#include <mpi.h>

Timer::Timer()
{
    start();
}

void Timer::start()
{
    MPI_Barrier(MPI_COMM_WORLD);
    st = std::chrono::high_resolution_clock::now();
}

void Timer::stop_and_log(char const *label)
{
    ed = std::chrono::high_resolution_clock::now();
    elapsed = ed - st;
    t = elapsed.count();

    double max_elapsed, avg_elapsed;
    MPI_Reduce(&t, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);    
    MPI_Reduce(&t, &avg_elapsed, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    int world_size, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    avg_elapsed /= world_size;

    if (myrank == 0) {
        std::cout << "[Timer] " << label << ":\n";
        std::cout  << "    time elapsed : " << std::fixed << std::setprecision(3) << max_elapsed << "s (max), " \
                    << avg_elapsed << "s (avg)\n\n" << std::flush;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}