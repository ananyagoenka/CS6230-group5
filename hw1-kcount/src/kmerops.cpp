#include "kmerops.hpp"
#include "dnaseq.hpp"
#include "timer.hpp"
#include <cstring>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <unordered_map>
#include <omp.h>
#include <unordered_map>
#include <vector>

// ForeachKmer is a function that takes a DnaBuffer object and a KmerHandler object.
// It iterates over the reads in the DnaBuffer object, extracts kmers from the reads,
// and calls the KmerHandler object for each kmer.
template <typename KmerHandler>
void ForeachKmer(const DnaBuffer &myreads, KmerHandler &handler)
{
    for (size_t i = 0; i < myreads.size(); ++i)
    {
        if (myreads[i].size() < KMER_SIZE)
            continue;

        std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);

        for (auto meritr = repmers.begin(); meritr != repmers.end(); ++meritr)
        {
            handler(*meritr);
        }
    }
}

// This function takes a kmer and the number of tasks and returns the owner of the kmer.
// The owner is calculated by hashing the kmer and then dividing the hash by the number of tasks.
// It is not used in the serial version of count_kmer,
// However it may come in handy when parallelizing the code.
int GetKmerOwner(const TKmer &kmer, int ntasks)
{
    uint64_t myhash = kmer.GetHash();
    double range = static_cast<double>(myhash) * static_cast<double>(ntasks);
    size_t owner = range / std::numeric_limits<uint64_t>::max();
    assert(owner >= 0 && owner < static_cast<size_t>(ntasks));
    return static_cast<int>(owner);
}

// This function takes a DnaBuffer object, counts the kmers in it, and returns a KmerList object.
// The current implementation is sequential and only uses one thread.
// Your task is to parallelize this function using OpenMP.
std::unique_ptr<KmerList> count_kmer_sort(const DnaBuffer &myreads)
{
    Timer timer;

    // Step1: Parse the kmers from the reads
    KmerSeedBucket *kmerseeds = new KmerSeedBucket;
    KmerParserHandler handler(*kmerseeds);
    ForeachKmer(myreads, handler);

    // Step2: Sort the kmers
    std::sort(kmerseeds->begin(), kmerseeds->end());

    // Step3: Count the kmers
    uint64_t valid_kmer = 0;
    KmerList *kmerlist = new KmerList();

    TKmer last_mer = (*kmerseeds)[0].kmer;
    uint64_t cur_kmer_cnt = 1;

    for (size_t idx = 1; idx < (*kmerseeds).size(); idx++)
    {
        TKmer cur_mer = (*kmerseeds)[idx].kmer;
        if (cur_mer == last_mer)
        {
            cur_kmer_cnt++;
        }
        else
        {
            // the next kmer has different value from the current one
            kmerlist->push_back(KmerListEntry());
            KmerListEntry &entry = kmerlist->back();
            TKmer &kmer = std::get<0>(entry);
            int &count = std::get<1>(entry);

            count = cur_kmer_cnt;
            kmer = last_mer;
            valid_kmer++;

            cur_kmer_cnt = 1;
            last_mer = cur_mer;
        }
    }

    // deal with the last kmer
    kmerlist->push_back(KmerListEntry());
    KmerListEntry &entry = kmerlist->back();
    TKmer &kmer = std::get<0>(entry);
    int &count = std::get<1>(entry);

    count = cur_kmer_cnt;
    kmer = last_mer;
    valid_kmer++;

    // Step4: Clean up
    delete kmerseeds;

    return std::unique_ptr<KmerList>(kmerlist);
}

// Another implementation using unordered_map
std::unique_ptr<KmerList> count_kmer(const DnaBuffer &myreads)
{
    Timer timer;

    std::unordered_map<KmerSeedStruct, int, KmerSeedHash> kmermap;
    KmerHashmapHandler handler(&kmermap);
    ForeachKmer(myreads, handler);

    KmerList *kmerlist = new KmerList();
    for (auto &entry : kmermap)
    {
        auto kmer = std::get<0>(entry);
        auto count = std::get<1>(entry);
        kmerlist->push_back(KmerListEntry(kmer.kmer, count));
    }

    return std::unique_ptr<KmerList>(kmerlist);
}

std::unique_ptr<KmerList> count_kmer_omp(const DnaBuffer &myreads)
{
    int num_threads = 0;
#pragma omp parallel
    {
#pragma omp single
        {
            num_threads = omp_get_num_threads();
        }
    }

    std::vector<std::unordered_map<TKmer, int>> local_maps(num_threads);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < myreads.size(); i++)
    {
        int tid = omp_get_thread_num();
        const DnaSeq &seq = myreads[i];
        if (seq.size() < KMER_SIZE)
            continue;

        auto rep_kmers = TKmer::GetRepKmers(seq);
        for (auto &kmer : rep_kmers)
        {
            local_maps[tid][kmer]++;
        }
    }

    std::unordered_map<TKmer, int> global_map;
    for (int t = 0; t < num_threads; t++)
    {
        for (auto &entry : local_maps[t])
        {
            global_map[entry.first] += entry.second;
        }
    }

    KmerList *kmerlist = new KmerList();
    for (auto &entry : global_map)
    {
        kmerlist->push_back(KmerListEntry(entry.first, entry.second));
    }

    return std::unique_ptr<KmerList>(kmerlist);
}