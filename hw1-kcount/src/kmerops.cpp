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

// std::unique_ptr<KmerList> count_kmer_omp(const DnaBuffer &myreads)
// {
//     int num_threads = 0;
// #pragma omp parallel
//     {
// #pragma omp single
//         {
//             num_threads = omp_get_num_threads();
//         }
//     }

//     std::vector<std::unordered_map<TKmer, int>> local_maps(num_threads);

// #pragma omp parallel for schedule(dynamic)
//     for (size_t i = 0; i < myreads.size(); i++)
//     {
//         int tid = omp_get_thread_num();
//         const DnaSeq &seq = myreads[i];
//         if (seq.size() < KMER_SIZE)
//             continue;

//         auto rep_kmers = TKmer::GetRepKmers(seq);
//         for (auto &kmer : rep_kmers)
//         {
//             local_maps[tid][kmer]++;
//         }
//     }

//     std::unordered_map<TKmer, int> global_map;
//     for (int t = 0; t < num_threads; t++)
//     {
//         for (auto &entry : local_maps[t])
//         {
//             global_map[entry.first] += entry.second;
//         }
//     }

//     KmerList *kmerlist = new KmerList();
//     for (auto &entry : global_map)
//     {
//         kmerlist->push_back(KmerListEntry(entry.first, entry.second));
//     }

//     return std::unique_ptr<KmerList>(kmerlist);
// }


// std::unique_ptr<KmerList> count_kmer_omp(const DnaBuffer& myreads)
// {
//     Timer timer;
    
//     // Get the number of available threads
//     int num_threads = omp_get_max_threads();
    
//     // Create thread-local hash maps to avoid contention
//     std::vector<std::unordered_map<KmerSeedStruct, int, KmerSeedHash>> thread_kmermaps(num_threads);
    
//     // Step 1: Parallel k-mer counting
//     #pragma omp parallel
//     {
//         int thread_id = omp_get_thread_num();
//         auto& local_kmermap = thread_kmermaps[thread_id];
        
//         // Process reads in parallel
//         #pragma omp for schedule(dynamic, 64)
//         for (size_t i = 0; i < myreads.size(); ++i)
//         {
//             if (myreads[i].size() < KMER_SIZE)
//                 continue;
                
//             std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);
            
//             // Process k-mers from this read
//             for (auto& kmer : repmers)
//             {
//                 KmerSeedStruct kmerseed(kmer);
//                 auto it = local_kmermap.find(kmerseed);
//                 if (it == local_kmermap.end()) {
//                     local_kmermap[kmerseed] = 1;
//                 } else {
//                     it->second++;
//                 }
//             }
//         }
//     }
    
//     // Step 2: Merge thread-local results
//     // Use a parallel reduction approach to minimize contention
    
//     // First merge pairs of thread maps in parallel
//     for (int stride = 1; stride < num_threads; stride *= 2) {
//         #pragma omp parallel for schedule(dynamic, 1)
//         for (int i = 0; i < num_threads; i += stride * 2) {
//             if (i + stride < num_threads) {
//                 // Merge the smaller map into the larger one for efficiency
//                 auto& map1 = thread_kmermaps[i];
//                 auto& map2 = thread_kmermaps[i + stride];
                
//                 if (map1.size() < map2.size()) {
//                     // Merge map1 into map2
//                     for (auto& entry : map1) {
//                         auto it = map2.find(entry.first);
//                         if (it == map2.end()) {
//                             map2[entry.first] = entry.second;
//                         } else {
//                             it->second += entry.second;
//                         }
//                     }
//                     // Clear map1 to free memory
//                     map1.clear();
//                 } else {
//                     // Merge map2 into map1
//                     for (auto& entry : map2) {
//                         auto it = map1.find(entry.first);
//                         if (it == map1.end()) {
//                             map1[entry.first] = entry.second;
//                         } else {
//                             it->second += entry.second;
//                         }
//                     }
//                     // Clear map2 to free memory
//                     map2.clear();
//                 }
//             }
//         }
//         // Synchronize before the next merge level
//         #pragma omp barrier
//     }
    
//     // Final result is in thread_kmermaps[0]
//     auto& final_kmermap = thread_kmermaps[0];
    
//     // Step 3: Create the KmerList from the final map
//     KmerList* kmerlist = new KmerList();
//     kmerlist->reserve(final_kmermap.size());
    
//     for (auto& entry : final_kmermap) {
//         kmerlist->push_back(KmerListEntry(entry.first.kmer, entry.second));
//     }
    
//     return std::unique_ptr<KmerList>(kmerlist);
// }

std::unique_ptr<KmerList> count_kmer_omp(const DnaBuffer& myreads)
{
    Timer timer;
    
    // Step 1: Parse and collect kmers in parallel
    KmerSeedBucket* kmerseeds = new KmerSeedBucket;
    
    // Estimate the number of k-mers based on the reads
    size_t total_bases = 0;
    for (size_t i = 0; i < myreads.size(); ++i) {
        total_bases += myreads[i].size();
    }
    // Reserve space to avoid frequent reallocations
    kmerseeds->reserve(total_bases > KMER_SIZE ? total_bases - KMER_SIZE + 1 : 0);
    
    // Thread-local vectors to avoid contention during collection
    int num_threads = omp_get_max_threads();
    std::vector<KmerSeedBucket> thread_kmerseeds(num_threads);
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto& local_kmerseeds = thread_kmerseeds[thread_id];
        
        // Process reads in parallel
        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < myreads.size(); ++i)
        {
            if (myreads[i].size() < KMER_SIZE)
                continue;
                
            std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);
            local_kmerseeds.reserve(local_kmerseeds.size() + repmers.size());
            
            for (auto& kmer : repmers) {
                local_kmerseeds.emplace_back(kmer);
            }
        }
    }
    
    // Merge all thread-local vectors into the main vector
    size_t total_size = 0;
    for (auto& local_seeds : thread_kmerseeds) {
        total_size += local_seeds.size();
    }
    kmerseeds->reserve(total_size);
    
    for (auto& local_seeds : thread_kmerseeds) {
        kmerseeds->insert(kmerseeds->end(), local_seeds.begin(), local_seeds.end());
        // Clear to free memory
        local_seeds.clear();
    }
    
    // Step 2: Sort the kmers in parallel
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::sort(kmerseeds->begin(), kmerseeds->end());
        }
    }
    
    // Step 3: Count the kmers
    KmerList* kmerlist = new KmerList();
    kmerlist->reserve(kmerseeds->size() / 2); // Estimate - we expect some repeats
    
    if (kmerseeds->empty()) {
        delete kmerseeds;
        return std::unique_ptr<KmerList>(kmerlist);
    }
    
    TKmer last_mer = (*kmerseeds)[0].kmer;
    uint64_t cur_kmer_cnt = 1;
    
    // This part is sequential as it depends on the order of sorted k-mers
    for (size_t idx = 1; idx < kmerseeds->size(); idx++) {
        TKmer cur_mer = (*kmerseeds)[idx].kmer;
        if (cur_mer == last_mer) {
            cur_kmer_cnt++;
        } else {
            // The next kmer has different value from the current one
            kmerlist->push_back(KmerListEntry(last_mer, cur_kmer_cnt));
            cur_kmer_cnt = 1;
            last_mer = cur_mer;
        }
    }
    
    // Deal with the last kmer
    kmerlist->push_back(KmerListEntry(last_mer, cur_kmer_cnt));
    
    // Step 4: Clean up
    delete kmerseeds;
    
    return std::unique_ptr<KmerList>(kmerlist);
}


// std::unique_ptr<KmerList> count_kmer_mpi(const DnaBuffer &myreads)
// {
//     Timer timer;
//     int world_size, myrank;
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

//     std::unordered_map<KmerSeedStruct, int, KmerSeedHash> local_kmer_map;
//     KmerHashmapHandler handler(&local_kmer_map);
//     ForeachKmer(myreads, handler);

//     std::vector<std::pair<KmerSeedStruct, int>> local_kmer_list(local_kmer_map.begin(), local_kmer_map.end());

//     int local_size = local_kmer_list.size();
//     std::vector<int> recv_sizes(world_size);
//     MPI_Gather(&local_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

//     std::vector<int> displs(world_size, 0);
//     int total_size = 0;
//     if (myrank == 0)
//     {
//         for (int i = 0; i < world_size; ++i)
//         {
//             displs[i] = total_size;
//             total_size += recv_sizes[i];
//         }
//     }

//     std::vector<std::pair<KmerSeedStruct, int>> global_kmer_list(total_size);
//     MPI_Gatherv(local_kmer_list.data(), local_size * sizeof(std::pair<KmerSeedStruct, int>), MPI_BYTE,
//                 global_kmer_list.data(), recv_sizes.data(), displs.data(), MPI_BYTE,
//                 0, MPI_COMM_WORLD);

//     std::unique_ptr<KmerList> kmerlist;
//     if (myrank == 0)
//     {
//         std::unordered_map<KmerSeedStruct, int, KmerSeedHash> global_kmer_map;
//         for (const auto &entry : global_kmer_list)
//         {
//             global_kmer_map[entry.first] += entry.second;
//         }

//         kmerlist = std::make_unique<KmerList>();
//         for (const auto &entry : global_kmer_map)
//         {
//             kmerlist->push_back(KmerListEntry(entry.first.kmer, entry.second));
//         }
//     }

//     MPI_Barrier(MPI_COMM_WORLD);
//     return kmerlist;
// }

// Add this to kmerops.cpp file

// MPI version of count_kmer
std::unique_ptr<KmerList> count_kmer_mpi(const DnaBuffer& myreads)
{
    Timer timer;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Step 1: Count local k-mers using a hash map
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash> local_kmermap;
    KmerHashmapHandler local_handler(&local_kmermap);
    ForeachKmer(myreads, local_handler);

    // Step 2: Determine which process should own each k-mer
    std::vector<std::unordered_map<KmerSeedStruct, int, KmerSeedHash>> send_kmermaps(size);

    // Distribute k-mers to their owner processes
    for (auto& entry : local_kmermap)
    {
        auto kmer = std::get<0>(entry);
        auto count = std::get<1>(entry);
        
        // Determine the owner of this k-mer based on its hash
        int owner = GetKmerOwner(kmer.kmer, size);
        
        // Add to the appropriate send map
        auto it = send_kmermaps[owner].find(kmer);
        if (it == send_kmermaps[owner].end()) {
            send_kmermaps[owner][kmer] = count;
        } else {
            it->second += count;
        }
    }

    // Clear the local map to free memory
    local_kmermap.clear();

    // Step 3: Exchange k-mers between processes
    // First, determine message sizes
    std::vector<int> send_sizes(size, 0);
    for (int i = 0; i < size; i++) {
        send_sizes[i] = send_kmermaps[i].size();
    }

    std::vector<int> recv_sizes(size, 0);
    MPI_Alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Prepare send buffers
    std::vector<std::vector<char>> send_buffers(size);
    for (int i = 0; i < size; i++) {
        if (send_sizes[i] > 0) {
            // Calculate buffer size: each entry has a TKmer and an int count
            size_t buffer_size = send_sizes[i] * (sizeof(TKmer) + sizeof(int));
            send_buffers[i].resize(buffer_size);
            
            char* ptr = send_buffers[i].data();
            for (const auto& entry : send_kmermaps[i]) {
                // Copy TKmer
                memcpy(ptr, &entry.first.kmer, sizeof(TKmer));
                ptr += sizeof(TKmer);
                
                // Copy count
                memcpy(ptr, &entry.second, sizeof(int));
                ptr += sizeof(int);
            }
        }
    }
    
    // Clear send maps to free memory
    send_kmermaps.clear();

    // Calculate send and receive displacements
    std::vector<int> send_displs(size, 0);
    std::vector<int> recv_displs(size, 0);
    std::vector<int> send_bytes(size, 0);
    std::vector<int> recv_bytes(size, 0);

    for (int i = 0; i < size; i++) {
        send_bytes[i] = send_buffers[i].size();
    }

    // Exchange buffer sizes
    MPI_Alltoall(send_bytes.data(), 1, MPI_INT, recv_bytes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate displacements
    for (int i = 1; i < size; i++) {
        send_displs[i] = send_displs[i-1] + send_bytes[i-1];
        recv_displs[i] = recv_displs[i-1] + recv_bytes[i-1];
    }

    // Allocate receive buffer
    size_t total_recv_bytes = 0;
    for (int i = 0; i < size; i++) {
        total_recv_bytes += recv_bytes[i];
    }
    
    std::vector<char> recv_buffer(total_recv_bytes);

    // Pack all send buffers into a single buffer
    std::vector<char> send_buffer_all;
    size_t total_send_bytes = 0;
    for (int i = 0; i < size; i++) {
        total_send_bytes += send_buffers[i].size();
    }
    send_buffer_all.resize(total_send_bytes);
    
    char* send_ptr = send_buffer_all.data();
    for (int i = 0; i < size; i++) {
        if (!send_buffers[i].empty()) {
            memcpy(send_ptr, send_buffers[i].data(), send_buffers[i].size());
            send_ptr += send_buffers[i].size();
        }
    }
    
    // Exchange data
    MPI_Alltoallv(
        send_buffer_all.data(), send_bytes.data(), send_displs.data(), MPI_BYTE,
        recv_buffer.data(), recv_bytes.data(), recv_displs.data(), MPI_BYTE,
        MPI_COMM_WORLD
    );
    
    // Clear send buffers to free memory
    send_buffers.clear();
    send_buffer_all.clear();

    // Step 4: Process received data
    std::unordered_map<KmerSeedStruct, int, KmerSeedHash> final_kmermap;
    
    char* recv_ptr = recv_buffer.data();
    for (int i = 0; i < size; i++) {
        size_t bytes_to_process = recv_bytes[i];
        char* end_ptr = recv_ptr + bytes_to_process;
        
        while (recv_ptr < end_ptr) {
            // Extract TKmer
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
    }

    // Step 5: Build the final KmerList
    KmerList* kmerlist = new KmerList();
    kmerlist->reserve(final_kmermap.size());
    
    for (const auto& entry : final_kmermap) {
        kmerlist->push_back(KmerListEntry(entry.first.kmer, entry.second));
    }

    return std::unique_ptr<KmerList>(kmerlist);
}