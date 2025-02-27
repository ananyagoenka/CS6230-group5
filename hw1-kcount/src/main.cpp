#include <iostream>
#include <iomanip>
#include <sstream>
#include <mpi.h>

#include "timer.hpp"
#include "fastaindex.hpp"
#include "dnabuffer.hpp"
#include "dnaseq.hpp"
#include "kmerops.hpp"

std::string fasta_fname;

void print_kmer_histogram(const KmerList &kmerlist)
{
    int maxcount = std::accumulate(kmerlist.cbegin(), kmerlist.cend(), 0, [](int cur, const auto &entry)
                                   { return std::max(cur, std::get<1>(entry)); });
    maxcount = std::min(maxcount, UPPERBOUND);
    int global_maxcount;
    MPI_Allreduce(&maxcount, &global_maxcount, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    std::vector<int> histo(global_maxcount + 1, 0);

    for (size_t i = 0; i < kmerlist.size(); ++i)
    {
        int cnt = std::get<1>(kmerlist[i]);
        assert(cnt >= 1);
        if (cnt > global_maxcount)
            continue;
        histo[cnt]++;
    }

    std::vector<int> histo_sum(histo.size(), 0);
    MPI_Reduce(histo.data(), histo_sum.data(), histo.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (myrank != 0)
        return;

    std::cout << "#count\tnumkmers" << std::endl;
    for (size_t i = 1; i < histo_sum.size(); ++i)
    {
        if (histo_sum[i] > 0)
        {
            std::cout << i << "\t" << histo_sum[i] << std::endl;
        }
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    Timer timer;
    std::ostringstream ss;

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <fasta file> [mode: omp|mpi|hybrid|serial]" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    fasta_fname = argv[1];

    timer.start();
    FastaIndex index(fasta_fname);
    ss << "reading " << std::quoted(index.get_faidx_fname()) << " and scattering to all MPI tasks";
    timer.stop_and_log(ss.str().c_str());
    ss.str("");
    ss.clear();

    timer.start();
    DnaBuffer mydna = index.getmydna();
    ss << "reading and 2-bit encoding " << std::quoted(index.get_fasta_fname()) << " sequences in parallel";
    timer.stop_and_log(ss.str().c_str());
    ss.str("");
    ss.clear();

    timer.start();
    std::unique_ptr<KmerList> kmerlist;
    if (argc >= 3)
    {
        std::string mode(argv[2]);
        if (mode == "omp")
        {
            kmerlist = count_kmer_omp(mydna);
        }
        else if (mode == "mpi")
        {
            kmerlist = count_kmer_mpi(mydna);
        }
        else if (mode == "hybrid")
        {
            kmerlist = count_kmer_hybrid(mydna);
        }
        else if (mode == "serial")
        {
            kmerlist = count_kmer(mydna);
        }
        else
        {
            std::cerr << "Unknown mode: " << mode << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    else
    {
        kmerlist = count_kmer(mydna);
    }
    timer.stop_and_log("Kmer Counting");

    print_kmer_histogram(*kmerlist);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}