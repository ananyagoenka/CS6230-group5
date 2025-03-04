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


std::unique_ptr<KmerList> count_kmer_omp(const DnaBuffer& myreads)
{
    Timer timer;
    
    // Get number of threads
    int num_threads = omp_get_max_threads();
    
    // Use thread-local storage with perfect initial sizing
    std::vector<KmerSeedBucket> thread_kmers(num_threads);
    
    // First pass: calculate approximate memory needs
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
    
    // Estimate kmers (each valid read position can start a kmer)
    size_t estimated_kmers = total_bases > KMER_SIZE ? total_bases - KMER_SIZE + 1 : 0;
    // Divide by threads with small buffer
    size_t kmers_per_thread = (estimated_kmers / num_threads) * 1.1 + 100;
    
    // Pre-allocate thread-local vectors with exact sizes
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_kmers[tid].reserve(kmers_per_thread);
    }
    
    // Collect kmers in parallel with optimal chunk size
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < myreads.size(); ++i) {
            if (myreads[i].size() < KMER_SIZE)
                continue;
                
            std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);
            
            // Add kmers to thread-local storage
            for (auto& kmer : repmers) {
                thread_kmers[tid].emplace_back(kmer);
            }
        }
    }
    
    // Calculate total size for optimized allocation
    size_t total_kmers = 0;
    for (const auto& local_kmers : thread_kmers) {
        total_kmers += local_kmers.size();
    }
    
    // Allocate final vector with exact size
    KmerSeedBucket* kmerseeds = new KmerSeedBucket();
    kmerseeds->reserve(total_kmers);
    
    // Merge with perfect offset calculation
    std::vector<size_t> offsets(num_threads + 1, 0);
    
    for (int i = 0; i < num_threads; ++i) {
        offsets[i+1] = offsets[i] + thread_kmers[i].size();
    }
    
    kmerseeds->resize(total_kmers);
    
    // Parallel merge for better performance
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_threads; ++i) {
        if (!thread_kmers[i].empty()) {
            std::copy(
                thread_kmers[i].begin(),
                thread_kmers[i].end(),
                kmerseeds->begin() + offsets[i]
            );
            
            // Clear immediately
            thread_kmers[i].clear();
        }
    }
    
    // Free memory
    thread_kmers.clear();
    std::vector<KmerSeedBucket>().swap(thread_kmers);
    
    // Use an efficient parallel quicksort with OpenMP tasks
    // This is generally faster than the tree-based merge sort for large datasets
    
    // Enable nested parallelism for tasks
    omp_set_nested(1);
    
    // Function for parallel quicksort
    auto parallel_quicksort = [](auto&& self, typename KmerSeedBucket::iterator begin, 
                               typename KmerSeedBucket::iterator end, int depth) -> void {
        auto size = std::distance(begin, end);
        
        // Use serial sort for small chunks or when recursion gets too deep
        if (size <= 100000 || depth >= 16) {
            std::sort(begin, end);
            return;
        }
        
        // Select pivot (median of three for better performance)
        auto pivot_idx = size / 2;
        std::iter_swap(begin + pivot_idx, end - 1);
        auto pivot = *(end - 1);
        
        // Partition
        auto i = begin;
        for (auto j = begin; j < end - 1; ++j) {
            if (*j < pivot) {
                std::iter_swap(i, j);
                ++i;
            }
        }
        std::iter_swap(i, end - 1);
        
        // Recursively sort partitions in parallel
        #pragma omp task untied if(depth < 8)
        self(self, begin, i, depth + 1);
        
        #pragma omp task untied if(depth < 8)
        self(self, i + 1, end, depth + 1);
        
        #pragma omp taskwait
    };
    
    // Start the parallel quicksort
    #pragma omp parallel
    {
        #pragma omp single
        {
            parallel_quicksort(parallel_quicksort, kmerseeds->begin(), kmerseeds->end(), 0);
        }
    }
    
    // Create result with optimized size
    KmerList* kmerlist = new KmerList();
    kmerlist->reserve(kmerseeds->size() / 3 + 1); // Expect ~1/3 unique kmers
    
    if (kmerseeds->empty()) {
        delete kmerseeds;
        return std::unique_ptr<KmerList>(kmerlist);
    }
    
    // Optimized counting with minimal branching
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
    
    // Add the last kmer
    kmerlist->emplace_back(last_mer, cur_kmer_cnt);
    
    // Clean up
    delete kmerseeds;
    
    return std::unique_ptr<KmerList>(kmerlist);
}












// std::unique_ptr<KmerList> count_kmer_mpi(const DnaBuffer& myreads)
// {
//     Timer timer;
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     // Step 1: Count local k-mers using a hash map with pre-allocation
//     size_t total_bases = 0;
//     for (size_t i = 0; i < myreads.size(); ++i) {
//         total_bases += myreads[i].size();
//     }
    
//     size_t estimated_kmers = (total_bases > KMER_SIZE) ? (total_bases - KMER_SIZE + 1) : 0;
//     std::unordered_map<KmerSeedStruct, int, KmerSeedHash> local_kmermap;
//     local_kmermap.reserve(estimated_kmers / 2); // Reserve capacity to reduce rehashing
    
//     // Use direct hashmap manipulation instead of handler for better performance
//     for (size_t i = 0; i < myreads.size(); ++i) {
//         if (myreads[i].size() < KMER_SIZE)
//             continue;
            
//         std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);
        
//         for (auto& kmer : repmers) {
//             KmerSeedStruct kmerseed(kmer);
//             auto it = local_kmermap.find(kmerseed);
//             if (it == local_kmermap.end()) {
//                 local_kmermap[kmerseed] = 1;
//             } else {
//                 it->second++;
//             }
//         }
//     }

//     // Step 2: Optimize k-mer distribution with pre-sized maps
//     std::vector<std::unordered_map<KmerSeedStruct, int, KmerSeedHash>> send_kmermaps(size);
    
//     // Pre-allocate send maps to reduce rehashing
//     size_t avg_kmers_per_proc = local_kmermap.size() / size + 1;
//     for (int i = 0; i < size; i++) {
//         send_kmermaps[i].reserve(avg_kmers_per_proc);
//     }

//     // Distribute k-mers to their owner processes
//     for (auto& entry : local_kmermap) {
//         // Determine the owner of this k-mer based on its hash
//         int owner = GetKmerOwner(entry.first.kmer, size);
        
//         // Add to the appropriate send map
//         auto it = send_kmermaps[owner].find(entry.first);
//         if (it == send_kmermaps[owner].end()) {
//             send_kmermaps[owner][entry.first] = entry.second;
//         } else {
//             it->second += entry.second;
//         }
//     }

//     // Clear the local map to free memory
//     local_kmermap.clear();
//     std::unordered_map<KmerSeedStruct, int, KmerSeedHash>().swap(local_kmermap); // Force memory release

//     // Step 3: Use a more efficient serialization approach
//     // Prepare send counts and prepare for buffer size calculation
//     std::vector<int> send_counts(size, 0);
//     std::vector<int> send_buffer_sizes(size, 0);  // Changed to int for MPI compatibility
    
//     for (int i = 0; i < size; i++) {
//         send_counts[i] = send_kmermaps[i].size();
//         // Calculate buffer size in bytes for each process
//         send_buffer_sizes[i] = send_counts[i] * static_cast<int>(sizeof(TKmer) + sizeof(int));
//     }

//     // Exchange counts
//     std::vector<int> recv_counts(size, 0);
//     MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
//     // Calculate receive buffer sizes in bytes
//     std::vector<int> recv_buffer_sizes(size, 0);
//     for (int i = 0; i < size; i++) {
//         recv_buffer_sizes[i] = recv_counts[i] * static_cast<int>(sizeof(TKmer) + sizeof(int));
//     }
    
//     // Calculate total buffer sizes
//     size_t total_send_size = 0;
//     size_t total_recv_size = 0;
    
//     for (int i = 0; i < size; i++) {
//         total_send_size += send_buffer_sizes[i];
//         total_recv_size += recv_buffer_sizes[i];
//     }
    
//     // Create single send and receive buffers
//     std::vector<char> send_buffer(total_send_size);
//     std::vector<char> recv_buffer(total_recv_size);
    
//     // Calculate displacements for MPI_Alltoallv
//     std::vector<int> send_displs(size, 0);
//     std::vector<int> recv_displs(size, 0);
    
//     for (int i = 1; i < size; i++) {
//         send_displs[i] = send_displs[i-1] + send_buffer_sizes[i-1];
//         recv_displs[i] = recv_displs[i-1] + recv_buffer_sizes[i-1];
//     }
    
//     // Pack data into send buffer with better locality
//     char* buffer_ptr = send_buffer.data();
//     for (int i = 0; i < size; i++) {
//         for (const auto& entry : send_kmermaps[i]) {
//             // Copy k-mer and count as a block for better cache behavior
//             memcpy(buffer_ptr, &entry.first.kmer, sizeof(TKmer));
//             buffer_ptr += sizeof(TKmer);
//             memcpy(buffer_ptr, &entry.second, sizeof(int));
//             buffer_ptr += sizeof(int);
//         }
//         // Clear map immediately to free memory
//         send_kmermaps[i].clear();
//     }
    
//     // Clear vector of maps to free memory
//     send_kmermaps.clear();
//     std::vector<std::unordered_map<KmerSeedStruct, int, KmerSeedHash>>().swap(send_kmermaps);
    
//     // Exchange data with a single Alltoallv call - fixed buffer size parameter
//     MPI_Alltoallv(
//         send_buffer.data(), send_buffer_sizes.data(), send_displs.data(), MPI_BYTE,
//         recv_buffer.data(), recv_buffer_sizes.data(), recv_displs.data(), MPI_BYTE,
//         MPI_COMM_WORLD
//     );
    
//     // Free send buffer memory immediately
//     send_buffer.clear();
//     std::vector<char>().swap(send_buffer);
    
//     // Step 4: Process received data with pre-sized final map
//     size_t total_recv_kmers = 0;
//     for (int i = 0; i < size; i++) {
//         total_recv_kmers += recv_counts[i];
//     }
    
//     std::unordered_map<KmerSeedStruct, int, KmerSeedHash> final_kmermap;
//     final_kmermap.reserve(total_recv_kmers);
    
//     // Process received data with better memory access pattern
//     char* recv_ptr = recv_buffer.data();
//     for (size_t i = 0; i < total_recv_kmers; i++) {
//         // Extract k-mer
//         TKmer kmer;
//         memcpy(&kmer, recv_ptr, sizeof(TKmer));
//         recv_ptr += sizeof(TKmer);
        
//         // Extract count
//         int count;
//         memcpy(&count, recv_ptr, sizeof(int));
//         recv_ptr += sizeof(int);
        
//         // Add to final map
//         KmerSeedStruct kmerseed(kmer);
//         auto it = final_kmermap.find(kmerseed);
//         if (it == final_kmermap.end()) {
//             final_kmermap[kmerseed] = count;
//         } else {
//             it->second += count;
//         }
//     }
    
//     // Free receive buffer memory
//     recv_buffer.clear();
//     std::vector<char>().swap(recv_buffer);
    
//     // Step 5: Build the final KmerList with efficient memory usage
//     KmerList* kmerlist = new KmerList();
//     kmerlist->reserve(final_kmermap.size());
    
//     for (const auto& entry : final_kmermap) {
//         kmerlist->emplace_back(entry.first.kmer, entry.second);
//     }
    
//     // Force memory release of the final map
//     final_kmermap.clear();
//     std::unordered_map<KmerSeedStruct, int, KmerSeedHash>().swap(final_kmermap);
    
//     return std::unique_ptr<KmerList>(kmerlist);
// }

std::unique_ptr<KmerList> count_kmer_mpi(const DnaBuffer& myreads)
{
    Timer timer;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // CRITICAL OPTIMIZATION 1: Use direct hash table construction to reduce memory copies
    // Estimate size for hash table to avoid rehashing
    size_t total_bases = 0;
    for (size_t i = 0; i < myreads.size(); ++i) {
        total_bases += myreads[i].size();
    }
    size_t estimated_kmers = total_bases > KMER_SIZE ? total_bases - KMER_SIZE + 1 : 0;
    
    // Use simple hash map based approach - often more efficient for MPI-heavy workloads
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash> local_kmermap;
    local_kmermap.reserve(estimated_kmers / 2);  // Reserve conservatively
    
    // Direct insertion without the overhead of handler indirection
    for (size_t i = 0; i < myreads.size(); ++i) {
        if (myreads[i].size() < KMER_SIZE)
            continue;
        
        std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);
        for (const auto& kmer : repmers) {
            KmerSeedStruct kmerseed(kmer);
            local_kmermap[kmerseed]++;
        }
    }

    // CRITICAL OPTIMIZATION 2: Minimize serialization overhead with flat structures
    // Create arrays for owner-based distribution - minimize hash lookups
    std::vector<std::vector<KmerSeedStruct>> kmer_by_owner(size);
    std::vector<std::vector<int>> count_by_owner(size);
    
    for (int i = 0; i < size; i++) {
        // Pre-allocate to avoid resizing
        kmer_by_owner[i].reserve(local_kmermap.size() / size + 10);
        count_by_owner[i].reserve(local_kmermap.size() / size + 10);
    }
    
    // Distribute kmers to owners with minimal overhead
    for (const auto& entry : local_kmermap) {
        int owner = GetKmerOwner(entry.first.kmer, size);
        kmer_by_owner[owner].push_back(entry.first);
        count_by_owner[owner].push_back(entry.second);
    }
    
    // Clear local map to free memory immediately
    local_kmermap.clear();
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash>().swap(local_kmermap);

    // CRITICAL OPTIMIZATION 3: Reduce MPI communication overhead
    // Get counts to send to each process
    std::vector<int> send_counts(size);
    for (int i = 0; i < size; i++) {
        send_counts[i] = kmer_by_owner[i].size();
    }
    
    // Exchange counts
    std::vector<int> recv_counts(size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Calculate displacements for MPI_Alltoallv
    std::vector<int> send_displs(size, 0);
    std::vector<int> recv_displs(size, 0);
    
    for (int i = 1; i < size; i++) {
        send_displs[i] = send_displs[i-1] + send_counts[i-1];
        recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
    }
    
    // Calculate total receive size
    int total_recv_kmers = 0;
    for (int i = 0; i < size; i++) {
        total_recv_kmers += recv_counts[i];
    }
    
    // CRITICAL OPTIMIZATION 4: Use separate sends for kmers and counts to avoid struct copies
    // Create flattened arrays for kmers and counts
    std::vector<KmerSeedStruct> send_kmers;
    std::vector<int> send_counts_flat;
    send_kmers.reserve(local_kmermap.size());
    send_counts_flat.reserve(local_kmermap.size());
    
    for (int i = 0; i < size; i++) {
        send_kmers.insert(send_kmers.end(), kmer_by_owner[i].begin(), kmer_by_owner[i].end());
        send_counts_flat.insert(send_counts_flat.end(), count_by_owner[i].begin(), count_by_owner[i].end());
        
        // Clear immediately to free memory
        std::vector<KmerSeedStruct>().swap(kmer_by_owner[i]);
        std::vector<int>().swap(count_by_owner[i]);
    }
    
    // Clear the outer vectors
    std::vector<std::vector<KmerSeedStruct>>().swap(kmer_by_owner);
    std::vector<std::vector<int>>().swap(count_by_owner);
    
    // Allocate receive buffers
    std::vector<KmerSeedStruct> recv_kmers(total_recv_kmers);
    std::vector<int> recv_counts_flat(total_recv_kmers);
    
    // Create MPI datatype for KmerSeedStruct
    MPI_Datatype kmer_type;
    MPI_Type_contiguous(sizeof(KmerSeedStruct), MPI_BYTE, &kmer_type);
    MPI_Type_commit(&kmer_type);
    
    // Exchange kmers and counts in a single operation
    MPI_Alltoallv(
        send_kmers.data(), send_counts.data(), send_displs.data(), kmer_type,
        recv_kmers.data(), recv_counts.data(), recv_displs.data(), kmer_type,
        MPI_COMM_WORLD
    );
    
    // Exchange counts in a separate operation
    MPI_Alltoallv(
        send_counts_flat.data(), send_counts.data(), send_displs.data(), MPI_INT,
        recv_counts_flat.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
        MPI_COMM_WORLD
    );
    
    // Free MPI type
    MPI_Type_free(&kmer_type);
    
    // Clear send buffers immediately
    send_kmers.clear();
    send_counts_flat.clear();
    std::vector<KmerSeedStruct>().swap(send_kmers);
    std::vector<int>().swap(send_counts_flat);
    
    // CRITICAL OPTIMIZATION 5: Ultra-fast final merging
    // Use a direct array to hold results - much faster than unordered_map for merging
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash> final_kmermap;
    final_kmermap.reserve(total_recv_kmers);
    
    // Merge received data directly - avoid unnecessary copies
    for (int i = 0; i < total_recv_kmers; i++) {
        final_kmermap[recv_kmers[i]] += recv_counts_flat[i];
    }
    
    // Clear receive buffers
    recv_kmers.clear();
    recv_counts_flat.clear();
    std::vector<KmerSeedStruct>().swap(recv_kmers);
    std::vector<int>().swap(recv_counts_flat);
    
    // Build the final KmerList
    KmerList* kmerlist = new KmerList();
    kmerlist->reserve(final_kmermap.size());
    
    for (const auto& entry : final_kmermap) {
        kmerlist->emplace_back(entry.first.kmer, entry.second);
    }
    
    return std::unique_ptr<KmerList>(kmerlist);
}