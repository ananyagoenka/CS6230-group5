#ifndef COMPILE_TIME_H_
#define COMPILE_TIME_H_

#include <limits>
#include <cstdint>

#ifndef KMER_SIZE
#error "KMER_SIZE must be defined"
#else
static_assert(2 < KMER_SIZE && KMER_SIZE < 96 && !!(KMER_SIZE & 1));
#ifdef SMER_SIZE
static_assert(0 < SMER_SIZE && SMER_SIZE <= KMER_SIZE);
#endif
#endif

#endif
