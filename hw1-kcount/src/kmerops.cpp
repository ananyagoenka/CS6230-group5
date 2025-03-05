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
    } else if (impl == "hybrid") {
        return count_kmer_hybrid(myreads);
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
        for (size_t i = 0; i < myreads.size(); ++i) {
            if (myreads[i].size() >= KMER_SIZE) {
                total_bases += myreads[i].size();
            }
        }
    }
    
    size_t estimated_kmers = total_bases > KMER_SIZE ? total_bases - KMER_SIZE + 1 : 0;
    size_t kmers_per_thread = (estimated_kmers / num_threads) * 1.1 + 100;
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_kmers[tid].reserve(kmers_per_thread);
    }
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < myreads.size(); ++i) {
            if (myreads[i].size() < KMER_SIZE)
                continue;
                
            std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);
            
            for (auto& kmer : repmers) {
                thread_kmers[tid].emplace_back(kmer);
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
    
    for (int i = 0; i < num_threads; ++i) {
        offsets[i+1] = offsets[i] + thread_kmers[i].size();
    }
    
    kmerseeds->resize(total_kmers);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_threads; ++i) {
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
        for (auto j = begin; j < end - 1; ++j) {
            if (*j < pivot) {
                std::iter_swap(i, j);
                ++i;
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
    for (size_t i = 0; i < myreads.size(); ++i) {
        total_bases += myreads[i].size();
    }
    size_t estimated_kmers = total_bases > KMER_SIZE ? total_bases - KMER_SIZE + 1 : 0;
    
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash> local_kmermap;
    local_kmermap.reserve(estimated_kmers / 2);  
    
    for (size_t i = 0; i < myreads.size(); ++i) {
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
    
    int total_recv_kmers = 0;
    for (int i = 0; i < size; i++) {
        total_recv_kmers += recv_counts[i];
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
    
    std::vector<KmerSeedStruct> recv_kmers(total_recv_kmers);
    std::vector<int> recv_counts_flat(total_recv_kmers);
    
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
    final_kmermap.reserve(total_recv_kmers);
    
    for (int i = 0; i < total_recv_kmers; i++) {
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


std::unique_ptr<KmerList> count_kmer_hybrid(const DnaBuffer& myreads)
{
    Timer timer;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Step 1: Count local k-mers using OpenMP parallelization for local processing
    // First, estimate the size of the hash map to avoid rehashing
    size_t total_bases = 0;
    
    #pragma omp parallel for reduction(+:total_bases)
    for (size_t i = 0; i < myreads.size(); ++i) {
        if (myreads[i].size() >= KMER_SIZE) {
            total_bases += myreads[i].size();
        }
    }
    
    size_t estimated_kmers = (total_bases > KMER_SIZE) ? (total_bases - KMER_SIZE + 1) / 2 : 0;
    
    // Create thread-local hash maps to reduce contention
    int num_threads = omp_get_max_threads();
    std::vector<std::unordered_map<KmerSeedStruct, int, KmerSeedHash>> thread_kmermaps(num_threads);
    
    // Pre-allocate thread-local maps
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_kmermaps[tid].reserve(estimated_kmers / num_threads);
    }
    
    // Count k-mers in parallel using OpenMP
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < myreads.size(); ++i) {
            if (myreads[i].size() < KMER_SIZE)
                continue;
                
            std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);
            
            for (auto& kmer : repmers) {
                KmerSeedStruct kmerseed(kmer);
                thread_kmermaps[tid][kmerseed]++;
            }
        }
    }
    
    // Step 2: Merge thread-local counts into a single local map
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash> local_kmermap;
    local_kmermap.reserve(estimated_kmers);
    
    for (int t = 0; t < num_threads; ++t) {
        for (const auto& entry : thread_kmermaps[t]) {
            local_kmermap[entry.first] += entry.second;
        }
        // Clear thread map immediately to free memory
        thread_kmermaps[t].clear();
    }
    
    // Clear thread_kmermaps vector to free memory
    thread_kmermaps.clear();
    std::vector<std::unordered_map<KmerSeedStruct, int, KmerSeedHash>>().swap(thread_kmermaps);

    // Step 3: Determine k-mer distribution using MPI
    // Pre-allocate maps for each MPI process to collect k-mers to send
    std::vector<std::unordered_map<KmerSeedStruct, int, KmerSeedHash>> send_kmermaps(size);
    
    // Estimate the number of k-mers per process
    size_t avg_kmers_per_proc = local_kmermap.size() / size + 1;
    for (int i = 0; i < size; i++) {
        send_kmermaps[i].reserve(avg_kmers_per_proc);
    }

    // Distribute k-mers to their owner processes
    for (const auto& entry : local_kmermap) {
        int owner = GetKmerOwner(entry.first.kmer, size);
        send_kmermaps[owner][entry.first] += entry.second;
    }
    
    // Clear local map to free memory
    local_kmermap.clear();
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash>().swap(local_kmermap);

    // Step 4: Pack data for MPI communication
    // Calculate send counts and buffer sizes
    std::vector<int> send_counts(size, 0);
    std::vector<int> send_buffer_sizes(size, 0);
    
    for (int i = 0; i < size; i++) {
        send_counts[i] = send_kmermaps[i].size();
        send_buffer_sizes[i] = send_counts[i] * static_cast<int>(sizeof(TKmer) + sizeof(int));
    }
    
    // Exchange counts between processes
    std::vector<int> recv_counts(size, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Calculate receive buffer sizes
    std::vector<int> recv_buffer_sizes(size, 0);
    for (int i = 0; i < size; i++) {
        recv_buffer_sizes[i] = recv_counts[i] * static_cast<int>(sizeof(TKmer) + sizeof(int));
    }
    
    // Calculate total buffer sizes and displacements
    size_t total_send_size = 0;
    size_t total_recv_size = 0;
    
    for (int i = 0; i < size; i++) {
        total_send_size += send_buffer_sizes[i];
        total_recv_size += recv_buffer_sizes[i];
    }
    
    // Create send and receive buffers
    std::vector<char> send_buffer(total_send_size);
    std::vector<char> recv_buffer(total_recv_size);
    
    // Calculate displacements for MPI_Alltoallv
    std::vector<int> send_displs(size, 0);
    std::vector<int> recv_displs(size, 0);
    
    for (int i = 1; i < size; i++) {
        send_displs[i] = send_displs[i-1] + send_buffer_sizes[i-1];
        recv_displs[i] = recv_displs[i-1] + recv_buffer_sizes[i-1];
    }
    
    // Pack data in parallel with OpenMP
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < size; i++) {
        char* buffer_ptr = send_buffer.data() + send_displs[i];
        for (const auto& entry : send_kmermaps[i]) {
            // Pack TKmer
            memcpy(buffer_ptr, &entry.first.kmer, sizeof(TKmer));
            buffer_ptr += sizeof(TKmer);
            
            // Pack count
            memcpy(buffer_ptr, &entry.second, sizeof(int));
            buffer_ptr += sizeof(int);
        }
        
        // Clear map immediately to free memory
        send_kmermaps[i].clear();
    }
    
    // Clear vector of maps to free memory
    send_kmermaps.clear();
    std::vector<std::unordered_map<KmerSeedStruct, int, KmerSeedHash>>().swap(send_kmermaps);
    
    // Step 5: Exchange data between processes
    MPI_Alltoallv(
        send_buffer.data(), send_buffer_sizes.data(), send_displs.data(), MPI_BYTE,
        recv_buffer.data(), recv_buffer_sizes.data(), recv_displs.data(), MPI_BYTE,
        MPI_COMM_WORLD
    );
    
    // Free send buffer memory immediately
    send_buffer.clear();
    std::vector<char>().swap(send_buffer);
    
    // Step 6: Process received data in parallel
    // Calculate total received k-mers
    size_t total_recv_kmers = 0;
    for (int i = 0; i < size; i++) {
        total_recv_kmers += recv_counts[i];
    }
    
    // Create a final hash map with exact sizing
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash> final_kmermap;
    final_kmermap.reserve(total_recv_kmers);
    
    // Divide processing into batches for better parallel performance
    const int batch_size = 10000;
    int num_batches = (total_recv_kmers + batch_size - 1) / batch_size;
    
    #pragma omp parallel
    {
        // Thread-local maps for batched processing
        std::unordered_map<KmerSeedStruct, int, KmerSeedHash> thread_map;
        thread_map.reserve(batch_size);
        
        #pragma omp for schedule(dynamic, 1)
        for (int batch = 0; batch < num_batches; batch++) {
            size_t start_idx = batch * batch_size;
            size_t end_idx = std::min(start_idx + batch_size, total_recv_kmers);
            size_t current_offset = 0;
            
            // Calculate the appropriate starting position in the buffer
            for (int i = 0; i < size; i++) {
                size_t process_start = recv_displs[i] / (sizeof(TKmer) + sizeof(int));
                size_t process_end = process_start + recv_counts[i];
                
                // If our batch overlaps with this process's data
                if (start_idx < process_end && end_idx > process_start) {
                    // Calculate the overlap
                    size_t overlap_start = std::max(start_idx, process_start);
                    size_t overlap_end = std::min(end_idx, process_end);
                    
                    // Process the overlap
                    char* data_ptr = recv_buffer.data() + recv_displs[i] + 
                                    (overlap_start - process_start) * (sizeof(TKmer) + sizeof(int));
                    
                    for (size_t j = overlap_start; j < overlap_end; j++) {
                        // Extract k-mer
                        TKmer kmer;
                        memcpy(&kmer, data_ptr, sizeof(TKmer));
                        data_ptr += sizeof(TKmer);
                        
                        // Extract count
                        int count;
                        memcpy(&count, data_ptr, sizeof(int));
                        data_ptr += sizeof(int);
                        
                        // Add to thread-local map
                        KmerSeedStruct kmerseed(kmer);
                        thread_map[kmerseed] += count;
                    }
                }
            }
            
            // Merge thread map into final map with minimal locking
            #pragma omp critical
            {
                for (const auto& entry : thread_map) {
                    final_kmermap[entry.first] += entry.second;
                }
            }
            
            // Clear thread map for next batch
            thread_map.clear();
        }
    }
    
    // Free receive buffer memory
    recv_buffer.clear();
    std::vector<char>().swap(recv_buffer);
    
    // Step 7: Build the final KmerList
    KmerList* kmerlist = new KmerList();
    kmerlist->reserve(final_kmermap.size());
    
    for (const auto& entry : final_kmermap) {
        kmerlist->emplace_back(entry.first.kmer, entry.second);
    }
    
    return std::unique_ptr<KmerList>(kmerlist);
}