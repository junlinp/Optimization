#include <immintrin.h>

int no_avx_reduce(int * arr, size_t N) {
    int sum = 0;
    for(int i = 0; i < N; i++) {
        sum += arr[i];
    }
    return sum;
}
int avx_reduce(int * arr, size_t N) {
    __m256i reduce_sum = _mm256_setzero_si256();
    for(int i = 0; i < N / 8; i++) {
        __m256i oprand = _mm256_load_si256((__m256i_u*)arr);
        reduce_sum = _mm256_add_epi32(reduce_sum, oprand);
        arr += 8;
    }

    int r[8]; 
    _mm256_store_si256((__m256i_u*)r, reduce_sum);
    return r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7];
}