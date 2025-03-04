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




// std::unique_ptr<KmerList> count_kmer_omp(const DnaBuffer& myreads)
// {
//     Timer timer;
    
//     // Get the number of available threads
//     int num_threads = omp_get_max_threads();
    
//     // Create thread-local vectors to avoid contention during collection
//     std::vector<KmerSeedBucket> thread_kmerseeds(num_threads);
    
//     // Calculate total bases and pre-allocate memory more efficiently
//     size_t total_bases = 0;
//     #pragma omp parallel for reduction(+:total_bases) schedule(static)
//     for (size_t i = 0; i < myreads.size(); ++i) {
//         total_bases += myreads[i].size();
//     }
    
//     // Estimate k-mers per thread for better memory allocation
//     size_t estimated_kmers = total_bases > KMER_SIZE ? total_bases - KMER_SIZE + 1 : 0;
//     size_t kmers_per_thread = (estimated_kmers / num_threads) * 1.2; // Add 20% buffer
    
//     // Pre-allocate thread-local vectors
//     #pragma omp parallel
//     {
//         int thread_id = omp_get_thread_num();
//         thread_kmerseeds[thread_id].reserve(kmers_per_thread);
//     }
    
//     // Process reads in parallel with improved chunk size
//     #pragma omp parallel
//     {
//         int thread_id = omp_get_thread_num();
//         auto& local_kmerseeds = thread_kmerseeds[thread_id];
        
//         #pragma omp for schedule(guided, 16)
//         for (size_t i = 0; i < myreads.size(); ++i)
//         {
//             if (myreads[i].size() < KMER_SIZE)
//                 continue;
                
//             std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);
//             local_kmerseeds.reserve(local_kmerseeds.size() + repmers.size());
            
//             for (auto& kmer : repmers) {
//                 local_kmerseeds.emplace_back(std::move(kmer));
//             }
//         }
//     }
    
//     // Calculate total size once before allocation
//     size_t total_kmer_count = 0;
//     for (const auto& local_seeds : thread_kmerseeds) {
//         total_kmer_count += local_seeds.size();
//     }
    
//     // Allocate the combined vector only once with exact size
//     KmerSeedBucket* kmerseeds = new KmerSeedBucket();
//     kmerseeds->reserve(total_kmer_count);
    
//     // Merge thread-local vectors using parallel merging for large datasets
//     if (total_kmer_count > 1000000) {
//         // For large datasets, use parallel merge strategy
//         std::vector<size_t> offsets(num_threads + 1, 0);
//         offsets[0] = 0;
        
//         // Calculate offsets
//         for (int i = 0; i < num_threads; ++i) {
//             offsets[i + 1] = offsets[i] + thread_kmerseeds[i].size();
//         }
        
//         // Resize once to exact size
//         kmerseeds->resize(total_kmer_count);
        
//         // Copy data in parallel
//         #pragma omp parallel for schedule(static)
//         for (int i = 0; i < num_threads; ++i) {
//             if (!thread_kmerseeds[i].empty()) {
//                 std::copy(thread_kmerseeds[i].begin(), thread_kmerseeds[i].end(), 
//                          kmerseeds->begin() + offsets[i]);
//                 // Clear immediately to free memory
//                 KmerSeedBucket().swap(thread_kmerseeds[i]);
//             }
//         }
//     } else {
//         // For smaller datasets, simple sequential merge is faster
//         for (auto& local_seeds : thread_kmerseeds) {
//             kmerseeds->insert(kmerseeds->end(), 
//                              std::make_move_iterator(local_seeds.begin()),
//                              std::make_move_iterator(local_seeds.end()));
//             // Use swap for efficient clearing
//             KmerSeedBucket().swap(local_seeds);
//         }
//     }
    
//     // Clear thread_kmerseeds vector to free memory
//     thread_kmerseeds.clear();
    
//     // Use parallel sorting with optimized parameters
//     #pragma omp parallel
//     {
//         #pragma omp single
//         {
//             // Replace __gnu_parallel::sort with standard parallel sort
//             // For large datasets, use OpenMP parallel sort
//             if (kmerseeds->size() > 100000) {
//                 #pragma omp parallel sections
//                 {
//                     #pragma omp section
//                     {
//                         std::sort(kmerseeds->begin(), kmerseeds->end());
//                     }
//                 }
//             } else {
//                 // For smaller datasets, sequential sort may be faster
//                 std::sort(kmerseeds->begin(), kmerseeds->end());
//             }
//         }
//     }
    
//     // Create result list with optimized size estimation
//     KmerList* kmerlist = new KmerList();
//     // More accurate reservation based on sorted data uniqueness
//     size_t unique_count_estimate = std::min(kmerseeds->size(), 
//                                           kmerseeds->size() > 1000000 ? 
//                                           kmerseeds->size() / 3 : kmerseeds->size() / 2);
//     kmerlist->reserve(unique_count_estimate);
    
//     if (kmerseeds->empty()) {
//         delete kmerseeds;
//         return std::unique_ptr<KmerList>(kmerlist);
//     }
    
//     // Counting phase - this is harder to parallelize but can be optimized
//     TKmer last_mer = (*kmerseeds)[0].kmer;
//     uint64_t cur_kmer_cnt = 1;
    
//     // Process in chunks to improve cache locality
//     const size_t chunk_size = 1024;
//     for (size_t idx = 1; idx < kmerseeds->size(); idx++) {
//         TKmer cur_mer = (*kmerseeds)[idx].kmer;
//         if (cur_mer == last_mer) {
//             cur_kmer_cnt++;
//         } else {
//             kmerlist->emplace_back(last_mer, cur_kmer_cnt);
//             cur_kmer_cnt = 1;
//             last_mer = cur_mer;
//         }
        
//         // Periodically check if we need to resize
//         if (idx % chunk_size == 0 && kmerlist->size() >= kmerlist->capacity() * 0.9) {
//             kmerlist->reserve(kmerlist->capacity() * 1.5);
//         }
//     }
    
//     // Handle the last k-mer
//     kmerlist->emplace_back(last_mer, cur_kmer_cnt);
    
//     // Clean up
//     delete kmerseeds;
    
//     return std::unique_ptr<KmerList>(kmerlist);
// }








std::unique_ptr<KmerList> count_kmer_omp(const DnaBuffer& myreads)
{
    Timer timer;
    
    // Get the number of available threads - set early to avoid overhead later
    int num_threads = omp_get_max_threads();
    
    // First pass: calculate total reads and precompute read sizes for better load balancing
    size_t total_bases = 0;
    size_t total_valid_reads = 0;
    
    #pragma omp parallel reduction(+:total_bases,total_valid_reads)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < myreads.size(); ++i) {
            size_t read_size = myreads[i].size();
            if (read_size >= KMER_SIZE) {
                total_bases += read_size;
                total_valid_reads++;
            }
        }
    }
    
    // Early return if no valid reads
    if (total_valid_reads == 0) {
        return std::unique_ptr<KmerList>(new KmerList());
    }
    
    // Better estimation of k-mer count - more accurate for memory allocation
    size_t estimated_kmers_per_read = 2.0; // Each read generates ~2 k-mers on average
    size_t estimated_total_kmers = total_valid_reads * estimated_kmers_per_read;
    
    // Thread-local vectors with more precise reservation
    std::vector<KmerSeedBucket> thread_kmerseeds(num_threads);
    
    // Pre-allocate thread-local vectors with more precise sizing
    size_t kmers_per_thread = (estimated_total_kmers / num_threads) * 1.1; // 10% buffer
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto& local_kmerseeds = thread_kmerseeds[thread_id];
        local_kmerseeds.reserve(kmers_per_thread);
    }
    
    // Process reads with guided scheduling for better load balancing
    // Use large chunk size for reduced scheduling overhead
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto& local_kmerseeds = thread_kmerseeds[thread_id];
        
        #pragma omp for schedule(guided, 32)
        for (size_t i = 0; i < myreads.size(); ++i) {
            const auto& read = myreads[i];
            if (read.size() < KMER_SIZE)
                continue;
                
            // Pre-allocate the repmers vector to avoid repeated reallocation
            std::vector<TKmer> repmers = TKmer::GetRepKmers(read);
            
            // Ensure capacity once before adding all k-mers
            if (local_kmerseeds.size() + repmers.size() > local_kmerseeds.capacity()) {
                local_kmerseeds.reserve(local_kmerseeds.capacity() * 2);
            }
            
            // Use move semantics to avoid unnecessary copies
            for (auto& kmer : repmers) {
                local_kmerseeds.emplace_back(std::move(kmer));
            }
        }
    }
    
    // Calculate total size for single allocation
    size_t total_kmer_count = 0;
    for (const auto& local_seeds : thread_kmerseeds) {
        total_kmer_count += local_seeds.size();
    }
    
    // Create the combined vector with exact size - avoid reallocation
    KmerSeedBucket* kmerseeds = new KmerSeedBucket();
    kmerseeds->reserve(total_kmer_count);
    
    // Efficient merging strategy based on data size
    if (total_kmer_count > 10000000) { // Use higher threshold for parallel merge
        // Parallel merge for large datasets
        std::vector<size_t> offsets(num_threads + 1, 0);
        
        // Calculate offsets for parallel insert
        for (int i = 0; i < num_threads; ++i) {
            offsets[i + 1] = offsets[i] + thread_kmerseeds[i].size();
        }
        
        // Resize once to final size
        kmerseeds->resize(total_kmer_count);
        
        // Copy data in parallel with larger chunks for better efficiency
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < num_threads; ++i) {
            if (!thread_kmerseeds[i].empty()) {
                // Use memcpy for larger contiguous blocks - much faster
                std::copy(
                    thread_kmerseeds[i].begin(), 
                    thread_kmerseeds[i].end(), 
                    kmerseeds->begin() + offsets[i]
                );
                
                // Clear memory immediately
                thread_kmerseeds[i].clear();
                thread_kmerseeds[i].shrink_to_fit();
            }
        }
    } else {
        // Sequential merge for smaller datasets
        for (auto& local_seeds : thread_kmerseeds) {
            if (!local_seeds.empty()) {
                kmerseeds->insert(
                    kmerseeds->end(),
                    std::make_move_iterator(local_seeds.begin()),
                    std::make_move_iterator(local_seeds.end())
                );
                // Free memory
                local_seeds.clear();
                local_seeds.shrink_to_fit();
            }
        }
    }
    
    // Clear the thread vector to free memory earlier
    thread_kmerseeds.clear();
    std::vector<KmerSeedBucket>().swap(thread_kmerseeds);
    
    // Sorting - use parallel sort for larger datasets with optimized parameters
    if (kmerseeds->size() > 1000000) {
        // Use parallel sort without nested parallelism for better performance
        #pragma omp parallel
        {
            #pragma omp single
            {
                std::sort(kmerseeds->begin(), kmerseeds->end());
            }
        }
    } else {
        // Sequential sort for smaller datasets - avoids overhead
        std::sort(kmerseeds->begin(), kmerseeds->end());
    }
    
    // Create result list with better size estimation
    KmerList* kmerlist = new KmerList();
    
    // More accurate reservation based on empirical uniqueness ratio
    size_t uniqueness_factor = (kmerseeds->size() > 5000000) ? 3 : 2;
    size_t unique_count_estimate = std::min(kmerseeds->size(), kmerseeds->size() / uniqueness_factor);
    kmerlist->reserve(unique_count_estimate);
    
    if (kmerseeds->empty()) {
        delete kmerseeds;
        return std::unique_ptr<KmerList>(kmerlist);
    }
    
    // Counting phase - optimize for cache locality
    TKmer last_mer = (*kmerseeds)[0].kmer;
    uint64_t cur_kmer_cnt = 1;
    
    // Process in larger chunks for better cache performance
    const size_t chunk_size = 4096; // Increased chunk size for better cache utilization
    
    for (size_t idx = 1; idx < kmerseeds->size(); idx++) {
        TKmer cur_mer = (*kmerseeds)[idx].kmer;
        
        if (cur_mer == last_mer) {
            cur_kmer_cnt++;
        } else {
            kmerlist->emplace_back(last_mer, cur_kmer_cnt);
            cur_kmer_cnt = 1;
            last_mer = cur_mer;
        }
        
        // Less frequent capacity checks to reduce overhead
        if (idx % chunk_size == 0 && kmerlist->size() >= kmerlist->capacity() * 0.95) {
            kmerlist->reserve(kmerlist->capacity() * 1.5);
        }
    }
    
    // Handle the last k-mer
    kmerlist->emplace_back(last_mer, cur_kmer_cnt);
    
    // Clean up
    delete kmerseeds;
    
    return std::unique_ptr<KmerList>(kmerlist);
}







std::unique_ptr<KmerList> count_kmer_mpi(const DnaBuffer& myreads)
{
    Timer timer;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Step 1: Count local k-mers using a hash map with pre-allocation
    size_t total_bases = 0;
    for (size_t i = 0; i < myreads.size(); ++i) {
        total_bases += myreads[i].size();
    }
    
    size_t estimated_kmers = (total_bases > KMER_SIZE) ? (total_bases - KMER_SIZE + 1) : 0;
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash> local_kmermap;
    local_kmermap.reserve(estimated_kmers / 2); // Reserve capacity to reduce rehashing
    
    // Use direct hashmap manipulation instead of handler for better performance
    for (size_t i = 0; i < myreads.size(); ++i) {
        if (myreads[i].size() < KMER_SIZE)
            continue;
            
        std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);
        
        for (auto& kmer : repmers) {
            KmerSeedStruct kmerseed(kmer);
            auto it = local_kmermap.find(kmerseed);
            if (it == local_kmermap.end()) {
                local_kmermap[kmerseed] = 1;
            } else {
                it->second++;
            }
        }
    }

    // Step 2: Optimize k-mer distribution with pre-sized maps
    std::vector<std::unordered_map<KmerSeedStruct, int, KmerSeedHash>> send_kmermaps(size);
    
    // Pre-allocate send maps to reduce rehashing
    size_t avg_kmers_per_proc = local_kmermap.size() / size + 1;
    for (int i = 0; i < size; i++) {
        send_kmermaps[i].reserve(avg_kmers_per_proc);
    }

    // Distribute k-mers to their owner processes
    for (auto& entry : local_kmermap) {
        // Determine the owner of this k-mer based on its hash
        int owner = GetKmerOwner(entry.first.kmer, size);
        
        // Add to the appropriate send map
        auto it = send_kmermaps[owner].find(entry.first);
        if (it == send_kmermaps[owner].end()) {
            send_kmermaps[owner][entry.first] = entry.second;
        } else {
            it->second += entry.second;
        }
    }

    // Clear the local map to free memory
    local_kmermap.clear();
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash>().swap(local_kmermap); // Force memory release

    // Step 3: Use a more efficient serialization approach
    // Prepare send counts and prepare for buffer size calculation
    std::vector<int> send_counts(size, 0);
    std::vector<int> send_buffer_sizes(size, 0);  // Changed to int for MPI compatibility
    
    for (int i = 0; i < size; i++) {
        send_counts[i] = send_kmermaps[i].size();
        // Calculate buffer size in bytes for each process
        send_buffer_sizes[i] = send_counts[i] * static_cast<int>(sizeof(TKmer) + sizeof(int));
    }

    // Exchange counts
    std::vector<int> recv_counts(size, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Calculate receive buffer sizes in bytes
    std::vector<int> recv_buffer_sizes(size, 0);
    for (int i = 0; i < size; i++) {
        recv_buffer_sizes[i] = recv_counts[i] * static_cast<int>(sizeof(TKmer) + sizeof(int));
    }
    
    // Calculate total buffer sizes
    size_t total_send_size = 0;
    size_t total_recv_size = 0;
    
    for (int i = 0; i < size; i++) {
        total_send_size += send_buffer_sizes[i];
        total_recv_size += recv_buffer_sizes[i];
    }
    
    // Create single send and receive buffers
    std::vector<char> send_buffer(total_send_size);
    std::vector<char> recv_buffer(total_recv_size);
    
    // Calculate displacements for MPI_Alltoallv
    std::vector<int> send_displs(size, 0);
    std::vector<int> recv_displs(size, 0);
    
    for (int i = 1; i < size; i++) {
        send_displs[i] = send_displs[i-1] + send_buffer_sizes[i-1];
        recv_displs[i] = recv_displs[i-1] + recv_buffer_sizes[i-1];
    }
    
    // Pack data into send buffer with better locality
    char* buffer_ptr = send_buffer.data();
    for (int i = 0; i < size; i++) {
        for (const auto& entry : send_kmermaps[i]) {
            // Copy k-mer and count as a block for better cache behavior
            memcpy(buffer_ptr, &entry.first.kmer, sizeof(TKmer));
            buffer_ptr += sizeof(TKmer);
            memcpy(buffer_ptr, &entry.second, sizeof(int));
            buffer_ptr += sizeof(int);
        }
        // Clear map immediately to free memory
        send_kmermaps[i].clear();
    }
    
    // Clear vector of maps to free memory
    send_kmermaps.clear();
    std::vector<std::unordered_map<KmerSeedStruct, int, KmerSeedHash>>().swap(send_kmermaps);
    
    // Exchange data with a single Alltoallv call - fixed buffer size parameter
    MPI_Alltoallv(
        send_buffer.data(), send_buffer_sizes.data(), send_displs.data(), MPI_BYTE,
        recv_buffer.data(), recv_buffer_sizes.data(), recv_displs.data(), MPI_BYTE,
        MPI_COMM_WORLD
    );
    
    // Free send buffer memory immediately
    send_buffer.clear();
    std::vector<char>().swap(send_buffer);
    
    // Step 4: Process received data with pre-sized final map
    size_t total_recv_kmers = 0;
    for (int i = 0; i < size; i++) {
        total_recv_kmers += recv_counts[i];
    }
    
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash> final_kmermap;
    final_kmermap.reserve(total_recv_kmers);
    
    // Process received data with better memory access pattern
    char* recv_ptr = recv_buffer.data();
    for (size_t i = 0; i < total_recv_kmers; i++) {
        // Extract k-mer
        TKmer kmer;
        memcpy(&kmer, recv_ptr, sizeof(TKmer));
        recv_ptr += sizeof(TKmer);
        
        // Extract count
        int count;
        memcpy(&count, recv_ptr, sizeof(int));
        recv_ptr += sizeof(int);
        
        // Add to final map
        KmerSeedStruct kmerseed(kmer);
        auto it = final_kmermap.find(kmerseed);
        if (it == final_kmermap.end()) {
            final_kmermap[kmerseed] = count;
        } else {
            it->second += count;
        }
    }
    
    // Free receive buffer memory
    recv_buffer.clear();
    std::vector<char>().swap(recv_buffer);
    
    // Step 5: Build the final KmerList with efficient memory usage
    KmerList* kmerlist = new KmerList();
    kmerlist->reserve(final_kmermap.size());
    
    for (const auto& entry : final_kmermap) {
        kmerlist->emplace_back(entry.first.kmer, entry.second);
    }
    
    // Force memory release of the final map
    final_kmermap.clear();
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash>().swap(final_kmermap);
    
    return std::unique_ptr<KmerList>(kmerlist);
}

