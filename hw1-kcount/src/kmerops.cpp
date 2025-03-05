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
#include <mpi.h>
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


std::unique_ptr<KmerList> count_kmer(const DnaBuffer &myreads)
{
    const char* impl_env = getenv("KMER_IMPL");
    std::string impl = impl_env ? impl_env : "serial";
    
    if (impl == "omp") {
        return count_kmer_omp(myreads);
    } else if (impl == "mpi") {
        return count_kmer_mpi(myreads);
    } else {
        
        // original count_kmer implementation
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
}


std::unique_ptr<KmerList> count_kmer_omp(const DnaBuffer& myreads)
{
    Timer timer;
    
    int num_threads = omp_get_max_threads();
    
    std::vector<KmerSeedBucket> thread_kmers(num_threads);
    
    size_t total_bases = 0;
    
    #pragma omp parallel reduction(+:total_bases)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < myreads.size(); i++) {
            if (myreads[i].size() >= KMER_SIZE) {
                total_bases += myreads[i].size();
            }
        }
    }
    
    size_t estimated_kmers;
    if (total_bases > KMER_SIZE) {
        estimated_kmers = total_bases - KMER_SIZE + 1;
    } else {
        estimated_kmers = 0;
    }

    size_t kmers_per_thread = (estimated_kmers / num_threads) * 1.1 + 100;
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        thread_kmers[thread_id].reserve(kmers_per_thread);
    }
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        
        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < myreads.size(); i++) {
            if (myreads[i].size() < KMER_SIZE)
                continue;
                
            std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);
            
            for (auto& kmer : repmers) {
                thread_kmers[thread_id].emplace_back(kmer);
            }
        }
    }
    
    size_t total_kmers = 0;
    for (const auto& local_kmers : thread_kmers) {
        total_kmers += local_kmers.size();
    }
    
    KmerSeedBucket* kmerseeds = new KmerSeedBucket();
    kmerseeds->reserve(total_kmers);
    
    std::vector<size_t> offsets(num_threads + 1, 0);
    
    for (int i = 0; i < num_threads; i++) {
        offsets[i+1] = offsets[i] + thread_kmers[i].size();
    }
    
    kmerseeds->resize(total_kmers);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_threads; i++) {
        if (!thread_kmers[i].empty()) {
            std::copy(
                thread_kmers[i].begin(),
                thread_kmers[i].end(),
                kmerseeds->begin() + offsets[i]
            );
            
            thread_kmers[i].clear();
        }
    }
    
    thread_kmers.clear();
    std::vector<KmerSeedBucket>().swap(thread_kmers);
    
    
    omp_set_nested(1);
    
    auto parallel_quicksort = [](auto&& self, typename KmerSeedBucket::iterator begin, 
                               typename KmerSeedBucket::iterator end, int depth) -> void {
        auto size = std::distance(begin, end);
        
        if (size <= 100000 || depth >= 16) {
            std::sort(begin, end);
            return;
        }
        
        auto pivot_idx = size / 2;
        std::iter_swap(begin + pivot_idx, end - 1);
        auto pivot = *(end - 1);
        
        auto i = begin;
        for (auto j = begin; j < end - 1; j++) {
            if (*j < pivot) {
                std::iter_swap(i, j);
                i++;
            }
        }
        std::iter_swap(i, end - 1);
        
        #pragma omp task untied if(depth < 8)
        self(self, begin, i, depth + 1);
        
        #pragma omp task untied if(depth < 8)
        self(self, i + 1, end, depth + 1);
        
        #pragma omp taskwait
    };
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            parallel_quicksort(parallel_quicksort, kmerseeds->begin(), kmerseeds->end(), 0);
        }
    }
    
    KmerList* kmerlist = new KmerList();
    kmerlist->reserve(kmerseeds->size() / 3 + 1);
    
    if (kmerseeds->empty()) {
        delete kmerseeds;
        return std::unique_ptr<KmerList>(kmerlist);
    }
    
    TKmer last_mer = (*kmerseeds)[0].kmer;
    uint64_t cur_kmer_cnt = 1;
    
    for (size_t idx = 1; idx < kmerseeds->size(); idx++) {
        TKmer cur_mer = (*kmerseeds)[idx].kmer;
        
        if (cur_mer == last_mer) {
            cur_kmer_cnt++;
        } else {
            kmerlist->emplace_back(last_mer, cur_kmer_cnt);
            cur_kmer_cnt = 1;
            last_mer = cur_mer;
        }
    }
    
    kmerlist->emplace_back(last_mer, cur_kmer_cnt);
    
    delete kmerseeds;
    
    return std::unique_ptr<KmerList>(kmerlist);
}


std::unique_ptr<KmerList> count_kmer_mpi(const DnaBuffer& myreads)
{
    Timer timer;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    size_t total_bases = 0;
    for (size_t i = 0; i < myreads.size(); i++) {
        total_bases += myreads[i].size();
    }

    size_t estimated_kmers;
    if (total_bases > KMER_SIZE) {
        estimated_kmers = total_bases - KMER_SIZE + 1;
    } else {
        estimated_kmers = 0;
    }

    
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash> local_kmermap;
    local_kmermap.reserve(estimated_kmers / 2);  
    
    for (size_t i = 0; i < myreads.size(); i++) {
        if (myreads[i].size() < KMER_SIZE)
            continue;
        
        std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);
        for (const auto& kmer : repmers) {
            KmerSeedStruct kmerseed(kmer);
            local_kmermap[kmerseed]++;
        }
    }
    
    std::vector<std::vector<KmerSeedStruct>> kmer_by_owner(size);
    std::vector<std::vector<int>> count_by_owner(size);
    
    for (int i = 0; i < size; i++) {
        kmer_by_owner[i].reserve(local_kmermap.size() / size + 10);
        count_by_owner[i].reserve(local_kmermap.size() / size + 10);
    }
    
    for (const auto& entry : local_kmermap) {
        int owner = GetKmerOwner(entry.first.kmer, size);
        kmer_by_owner[owner].push_back(entry.first);
        count_by_owner[owner].push_back(entry.second);
    }
    
    local_kmermap.clear();
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash>().swap(local_kmermap);

    
    std::vector<int> send_counts(size);
    for (int i = 0; i < size; i++) {
        send_counts[i] = kmer_by_owner[i].size();
    }
    
    std::vector<int> recv_counts(size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    std::vector<int> send_displs(size, 0);
    std::vector<int> recv_displs(size, 0);
    
    for (int i = 1; i < size; i++) {
        send_displs[i] = send_displs[i-1] + send_counts[i-1];
        recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
    }
    
    int total_received_kmers = 0;
    for (int i = 0; i < size; i++) {
        total_received_kmers += recv_counts[i];
    }
    
    
    std::vector<KmerSeedStruct> send_kmers;
    std::vector<int> send_counts_flat;
    send_kmers.reserve(local_kmermap.size());
    send_counts_flat.reserve(local_kmermap.size());
    
    for (int i = 0; i < size; i++) {
        send_kmers.insert(send_kmers.end(), kmer_by_owner[i].begin(), kmer_by_owner[i].end());
        send_counts_flat.insert(send_counts_flat.end(), count_by_owner[i].begin(), count_by_owner[i].end());
        
        std::vector<KmerSeedStruct>().swap(kmer_by_owner[i]);
        std::vector<int>().swap(count_by_owner[i]);
    }
    
    std::vector<std::vector<KmerSeedStruct>>().swap(kmer_by_owner);
    std::vector<std::vector<int>>().swap(count_by_owner);
    
    std::vector<KmerSeedStruct> recv_kmers(total_received_kmers);
    std::vector<int> recv_counts_flat(total_received_kmers);
    
    MPI_Datatype kmer_type;
    MPI_Type_contiguous(sizeof(KmerSeedStruct), MPI_BYTE, &kmer_type);
    MPI_Type_commit(&kmer_type);
    
    MPI_Alltoallv(
        send_kmers.data(), send_counts.data(), send_displs.data(), kmer_type,
        recv_kmers.data(), recv_counts.data(), recv_displs.data(), kmer_type,
        MPI_COMM_WORLD
    );
    
    MPI_Alltoallv(
        send_counts_flat.data(), send_counts.data(), send_displs.data(), MPI_INT,
        recv_counts_flat.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
        MPI_COMM_WORLD
    );
    
    MPI_Type_free(&kmer_type);
    
    send_kmers.clear();
    send_counts_flat.clear();
    std::vector<KmerSeedStruct>().swap(send_kmers);
    std::vector<int>().swap(send_counts_flat);
    
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash> final_kmermap;
    final_kmermap.reserve(total_received_kmers);
    
    for (int i = 0; i < total_received_kmers; i++) {
        final_kmermap[recv_kmers[i]] += recv_counts_flat[i];
    }
    
    recv_kmers.clear();
    recv_counts_flat.clear();
    std::vector<KmerSeedStruct>().swap(recv_kmers);
    std::vector<int>().swap(recv_counts_flat);
    
    KmerList* kmerlist = new KmerList();
    kmerlist->reserve(final_kmermap.size());
    
    for (const auto& entry : final_kmermap) {
        kmerlist->emplace_back(entry.first.kmer, entry.second);
    }
    
    return std::unique_ptr<KmerList>(kmerlist);
}