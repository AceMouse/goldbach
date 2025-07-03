#include <iostream>
#include <stdint.h>
#include "check_composite.cu"

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int32_t i32;
typedef int64_t i64;
typedef __int128_t i128;


constexpr u32 _low_primes[] = {3,5,7,11};
constexpr u32 mult_low_primes() {
    u32 res = 1;
    for (u32 a:_low_primes){
        res*=a;
    }
    return res;
}
constexpr u32 mult = mult_low_primes();
constexpr u8 _is_prime_maybe(u32 i){
    u8 res = 0;
    i+=mult*2;
    for (u32 p=i*8*2+1;p<(i+1)*8*2+1; p+=2){
        for (u32 a:_low_primes){
            res |=(p%a==0)<<((p%16)/2);
        }
    }
    return ~res;
}
constexpr u8 _trailing_zeroes(u8 d){
    u8 r = 0;
    for (int i = 0; i<8; i++) {
        r+=(d&((0xFF)>>i))==0;
    }
    return r;
}
template<u32 N>
struct B {
    constexpr B() : z() {
        for (auto i = 0; i < N; ++i)
            z[i] = _trailing_zeroes(i); 
    }
    u8 z[N];
};
template<u32 N>
struct A {
    constexpr A() : mb() {
        for (auto i = 0; i != N; ++i)
            mb[i] = _is_prime_maybe(i); 
    }
    u8 mb[N];
};
constexpr u32 low_sieve_len = 1+(mult*2+15)/16;
constexpr u32 trailing_len = 0xFF;
__device__
auto trailing_zeroes = B<trailing_len>();
__device__
auto low_sieve = A<low_sieve_len>();
__device__
bool is_prime_maybe(u64 i){
    u8 is_odd = i&1;
    u8 is_low_prime = (i==2) | (i==3)|(i==5)|(i==7)|(i==11);
    i %= mult*2;
    i/= 2;
    return (is_low_prime) || (is_odd & (low_sieve.mb[i/8]>>(i%8)));
}

__global__ void number_is_maybe_prime_kernel(const u64 starting_number, u64 *output_primes, bool *flag, u32 num_candidates) {
    int tid = threadIdx.x; // [0..12*numbers_per_block-1]
    int idx = blockIdx.x*blockDim.x+tid;
    if (idx >= num_candidates)
        return;

    u64 n = (starting_number+2*idx) | 1;
    n += (n == 1);
    if (is_prime_maybe(n)){
        output_primes[idx] = n;
        flag[idx] = true;
    }
}
__global__ void number_is_maybe_prime_kernel_count(const u64 starting_number, u64 *output_primes, bool *flag, u32 num_candidates, u32* count) {
    int tid = threadIdx.x; // [0..12*numbers_per_block-1]
    int idx = blockIdx.x*blockDim.x+tid;
    if (idx >= num_candidates)
        return;

    u64 n = (starting_number+2*idx) | 1;
    n += (n == 1);
    if (is_prime_maybe(n)){
        u32 outidx = atomicAdd(count,1);
        flag[outidx] = true;
        output_primes[outidx] = n;
    }
}
__constant__ u32 bases[12] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
__global__ void number_is_composite_kernel(u64* numbers, bool *flag, u32 num_candidates, u32 a) {
    int tid = threadIdx.x; // [0..12*numbers_per_block-1]
    int idx = blockIdx.x*blockDim.x+tid;

    if (idx >= num_candidates){
//        printf("idx %d >= %d\n",idx, num_candidates-1);
        return;
    } else if (!flag[idx]){
        return;
    }

    flag[idx] = false;
    u64 n = numbers[idx];
    u64 d = n - 1;
    int s = __ffsll(d) - 1;
    d >>= s;

    if (a >= n){
        for (u64 b:bases){
            if (n == b) {
                flag[idx] = true;
                return;
            }
        }
    }
    bool is_not_composite = !(check_composite_l(n, a, d, s)) ;
    flag[idx] = is_not_composite;
}
__global__ void number_is_composite_kernel_count(u64* numbers,u64* output_numbers, bool *oflag, u32* num_candidates, u32* count, u32 a) {
    int tid = threadIdx.x; // [0..12*numbers_per_block-1]
    int idx = blockIdx.x*blockDim.x+tid;

    if (idx >= *num_candidates){
//        printf("idx %d >= %d\n",idx, num_candidates-1);
        return;
    }
    
    u64 n = numbers[idx];
    if (a >= n){
        for (u64 b:bases){
            if (n == b) {
                u32 outidx = atomicAdd(count,1);
                oflag[outidx] = true;
                output_numbers[outidx] = n;
                return;
            }
        }
    }
    u64 d = n - 1;
    int s = __ffsll(d) - 1;
    d >>= s;
    bool is_not_composite = !(check_composite_l(n, a, d, s)) ;
    if (is_not_composite){
        u32 outidx = atomicAdd(count,1);
        oflag[outidx] = true;
        output_numbers[outidx] = n;
    }
}
__global__ void number_is_composite_kernel_count_wrap(u64* numbers,u64* output_numbers, bool *oflag, u32* num_candidates, u32* count, u32 a) {
    number_is_composite_kernel_count<<<(*num_candidates+255)/256, 256>>>(numbers,output_numbers, oflag,num_candidates,count, a);
}

constexpr __constant__ int active_threads_per_number = 12;
constexpr __constant__ int threads_per_number = 16;
//constexpr __constant__ int numbers_per_block = 64;
#ifndef numbers_per_block
#define numbers_per_block 2
#endif
/*constexpr __constant__ int threads_per_block = threads_per_number*numbers_per_block;
constexpr __constant__ int blocks1 = ((delta/2)/numbers_per_block);
constexpr __constant__ int blocks2 = ((delta)/numbers_per_block);*/
__global__ void miller_rabin_64_kernel(const u64 starting_number, u64 *output_primes, bool *flag, u32 *prime_count, u32 num_candidates, int direction) {
    int tid = threadIdx.x; // [0..12*numbers_per_block-1]
    int idx = blockIdx.x;


    if (idx >= num_candidates || tid%threads_per_number >= active_threads_per_number)
        return;

    bool is_not_composite = true;
    int num = tid/threads_per_number;
    int base = tid%threads_per_number;
    u64 n = starting_number+2*(idx*numbers_per_block+num);
    if (n == 0){
        if (starting_number <= 2 && num == 0){
            if ( base == 0 ) {
                int out_idx = atomicAdd(prime_count, 1);
                output_primes[out_idx] = 2;
                flag[out_idx] = true;
            }
        } else if (idx == 0 && starting_number <= 3 && num == 1) {
            if ( base == 0 ) {
                int out_idx = atomicAdd(prime_count, 1);
                output_primes[out_idx] = 3;
                flag[out_idx] = true;
            }
        }
        return;
    }
    n|=1;
//    printf("n=%ld (tid=%d, idx=%d)\n",n,tid,idx);
    if (!is_prime_maybe(n))
        return;

    // Write n-1 = d * 2^s
    u64 d = n - 1;
    int s = __ffsll(d) - 1;
    d >>= s;

    u32 a = bases[base];
    if (a >= n) return;  // base must be < n

    is_not_composite = !(check_composite(n, a, d, s)) ;


    // All threads must agree for primality
    constexpr u32 mask_threads = (1UL<<threads_per_number)-1;
    u32 mask = mask_threads<<(12*(tid%32>=threads_per_number));
    int all_pass = __all_sync(mask, is_not_composite);

    if (base == 0 && all_pass){
        int out_idx = atomicAdd(prime_count, 1);
        output_primes[out_idx] = n;
        flag[out_idx] = true;
    }
}
/*__global__ void miller_rabin_64_kernel(const u64 starting_number, u64 *output_primes, bool *flag, u32 *prime_count, u32 num_candidates, int direction) {
    int tid = threadIdx.x;       // [0..11]
    int idx = blockIdx.x;        // one number per block

    if (idx >= num_candidates || tid >= 24)
        return;

    bool is_not_composite = true;
    bool num_2 = tid>=12;
    u64 n = (starting_number+direction*idx*4 + num_2*2)|1;
    n += n==1;
    if (n < 4 || ((n & 1) == 0)) {
        if (tid == 12 || tid == 0 ) {
            if (n == 2 || n == 3){
                int out_idx = atomicAdd(prime_count, 1);
                output_primes[out_idx] = n;
                flag[out_idx] = true;
            }
        }
        return;
    }
    if (!is_prime_maybe(n))
        return;

    // Write n-1 = d * 2^s
    u64 d = n - 1;
    int s = __ffsll(d) - 1;
    d >>= s;

    u32 a = bases[tid-num_2*12];
    if (a >= n) return;  // base must be < n

    is_not_composite = !(check_composite(n, a, d, s)) ;


    // All threads must agree for primality
    unsigned mask = 0xFFF<<(12*num_2);  // lower 12 bits (threads 0..11)
    int all_pass = __all_sync(mask, is_not_composite);

    if ((tid==12 || tid == 0) && all_pass){
        int out_idx = atomicAdd(prime_count, 1);
        output_primes[out_idx] = n;
        flag[out_idx] = true;
    }
}*/
__global__ void goldbach_kernel(const u64 starting_number, const u64 ending_number, const u64* __restrict__ lo_primes, const u64* __restrict__ hi_primes, const u32* __restrict__ lo_prime_cnt, const u32* __restrict__ hi_prime_cnt) {
    int tid = threadIdx.x;       // [0..11]
    int wid = blockIdx.x;
    u64 idx = wid*blockDim.x;        // one number per block

    u64 n = starting_number+idx*2+tid*2;
    if (n >= ending_number)
        return;
//    printf("n=%ld\n",n);
    u32 lo_idx = 0;
    u32 hi_idx = (*hi_prime_cnt)-1;
    u64 lo = lo_primes[lo_idx];
    u64 hi = hi_primes[hi_idx];

    u64 s = hi+lo;
//    printf("s=%ld\n",s);
    while ( s != n ){
        lo_idx += (s < n);
        hi_idx -= (s > n);
        if (lo_idx == *lo_prime_cnt)
            break;
        lo = lo_primes[lo_idx];
        hi = hi_primes[hi_idx];
        s = hi+lo;
        if (hi_idx == 0 && (s > n))
            break;
/*        printf("lo[%d]=%ld\n",lo_idx,lo);
        printf("hi[%d]=%ld\n",hi_idx,hi);
        printf("s=%ld\n",s);*/
    }
    if (s != n) {
        bool lo_sorted = true;
        u64 prev = 0;
        for (int i = 0; i< *lo_prime_cnt; i++){
            printf("lo_primes[%d] = %ld\n",i,lo_primes[i]);
            lo_sorted |= prev < lo_primes[i];
            prev = lo_primes[i];
        }
        bool hi_sorted = true;
        prev = 0;
        for (int i = 0; i< *hi_prime_cnt; i++){
            printf("hi_primes[%d] = %ld\n",i,hi_primes[i]);
            lo_sorted |= prev < hi_primes[i];
            prev = hi_primes[i];
        }
        if (!lo_sorted)
            printf("lo_primes not sorted!\n");
        if (!hi_sorted)
            printf("hi_primes not sorted!\n");
        printf("%ld is a counter example\n",n);
    }
//    else printf("%ld = %ld + %ld\n",n, lo, hi);
}
__global__ void print_array(u32 *cnt, u64* array){
    bool sorted = true;
    u64 prev = 0;
    for (int i = 0; i< *cnt; i++){
        printf("array[%d] = %ld\n",i,array[i]);
        sorted |= prev < array[i];
        prev = array[i];
    }
    if (!sorted)
        printf("array is not sorted!\n");
}
__global__ void print_array(u32 cnt, bool* array){
    bool sorted = true;
    u64 prev = 0;
    for (int i = 0; i< cnt; i++){
        printf("array[%d] = %d\n",i,array[i]);
        sorted |= prev < array[i];
        prev = array[i];
    }
    if (!sorted)
        printf("array is not sorted!\n");
}
__global__ void print_arrays(u32 cnt, bool* flag, u64* array){
    bool sorted = true;
    u64 prev = 0;
    u32 trues = 0;
    for (int i = 0; i < cnt; i++){
        if (!flag[i]) continue;
        trues++;
        printf("array[%d] = %ld\n",i,array[i]);
        sorted |= prev < array[i];
        prev = array[i];
    }
    if (!sorted)
        printf("array is not sorted!\n");
    printf("array has %d trues!\n", trues);

}
#include <chrono>
#include <ctime>
std::string padZero(int num) {
    return (num < 10 ? "0" : "") + std::to_string(num);
}

std::string formatDuration(int years, int days, int hours, int minutes, int seconds) {
    std::string result;

    if (years > 0) {
        result += std::to_string(years) + "y ";
    }
    if (days > 0 || years > 0) {
        result += std::to_string(days) + "d ";
    }

    if (days > 0 || hours > 0 || years > 0) {
        result += padZero(hours) + "h ";
    }

    result += padZero(minutes) + "m ";
    result += padZero(seconds) + "s                              ";

    return result;
}
std::string format_duration(std::chrono::seconds secs) {
    int years = static_cast<int>(secs.count() / (86400*365));
    secs -= std::chrono::seconds(years * 86400 * 365);
    int days = static_cast<int>(secs.count() / 86400);
    secs -= std::chrono::seconds(days * 86400);
    int hours = static_cast<int>(secs.count() / 3600);
    secs -= std::chrono::seconds(hours * 3600);
    int minutes = static_cast<int>(secs.count() / 60);
    secs -= std::chrono::seconds(minutes * 60);
    int seconds = static_cast<int>(secs.count());


    return  formatDuration(years,days, hours, minutes,seconds);
}

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
int main(int argc, char** argv) {
    u64 delta = std::stod(argv[1]);
    u64 to_check = std::stod(argv[2]);
    cudaSetDevice(0);  // or whatever valid device you want
    time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "Starting calculation: " << ctime(&now) << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    u64 *d_primes_lo1;
    u64 *d_primes_hi1;
    u64 *d_primes_lo2;
    u64 *d_primes_hi2;
    bool *d_flags_lo1;
    bool *d_flags_hi1;
    bool *d_flags_lo2;
    bool *d_flags_hi2;
    u32 *lo_prime_cnts;
    u32 *hi_prime_cnts;
    cudaMalloc(&d_primes_lo1, delta * sizeof(u64));
    cudaMalloc(&d_primes_lo2, delta * sizeof(u64));
    cudaMalloc(&d_flags_lo1, delta * sizeof(bool));
    cudaMalloc(&d_flags_lo2, delta * sizeof(bool));
    cudaMalloc(&d_primes_hi1, delta * 2 * sizeof(u64));
    cudaMalloc(&d_primes_hi2, delta * 2 * sizeof(u64));
    cudaMalloc(&d_flags_hi1, delta * 2 * sizeof(bool));
    cudaMalloc(&d_flags_hi2, delta * 2 * sizeof(bool));

    cudaMalloc(&lo_prime_cnts, 13*sizeof(u32));
    cudaMalloc(&hi_prime_cnts, 13*sizeof(u32));
    thrust::device_ptr<u64> lo1_ptr = thrust::device_pointer_cast(d_primes_lo1);
    thrust::device_ptr<u64> lo2_ptr = thrust::device_pointer_cast(d_primes_lo2);
    thrust::device_ptr<u64> hi1_ptr = thrust::device_pointer_cast(d_primes_hi1);
    thrust::device_ptr<u64> hi2_ptr = thrust::device_pointer_cast(d_primes_hi2);
    thrust::device_ptr<bool> lo_flags1_ptr = thrust::device_pointer_cast(d_flags_lo1);
    thrust::device_ptr<bool> hi_flags1_ptr = thrust::device_pointer_cast(d_flags_hi1);
    thrust::device_ptr<bool> lo_flags2_ptr = thrust::device_pointer_cast(d_flags_lo2);
    thrust::device_ptr<bool> hi_flags2_ptr = thrust::device_pointer_cast(d_flags_hi2);

    cudaMemset(lo_prime_cnts, 0, 13*sizeof(u32));
    cudaMemset(d_flags_lo1, 0, delta*sizeof(bool));
    number_is_maybe_prime_kernel_count<<<(delta/2+255)/256, 256>>>(0, d_primes_lo1, d_flags_lo1, delta, &lo_prime_cnts[0]);
    number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_lo1, d_primes_lo2, d_flags_lo1, &lo_prime_cnts[0 ], &lo_prime_cnts[1 ], 2);
    number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_lo2, d_primes_lo1, d_flags_lo1, &lo_prime_cnts[1 ], &lo_prime_cnts[2 ], 3);
    number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_lo1, d_primes_lo2, d_flags_lo1, &lo_prime_cnts[2 ], &lo_prime_cnts[3 ], 5);
    number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_lo2, d_primes_lo1, d_flags_lo1, &lo_prime_cnts[3 ], &lo_prime_cnts[4 ], 7);
    number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_lo1, d_primes_lo2, d_flags_lo1, &lo_prime_cnts[4 ], &lo_prime_cnts[5 ], 11);
    number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_lo2, d_primes_lo1, d_flags_lo1, &lo_prime_cnts[5 ], &lo_prime_cnts[6 ], 13);
    number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_lo1, d_primes_lo2, d_flags_lo1, &lo_prime_cnts[6 ], &lo_prime_cnts[7 ], 17);
    number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_lo2, d_primes_lo1, d_flags_lo1, &lo_prime_cnts[7 ], &lo_prime_cnts[8 ], 19);
    number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_lo1, d_primes_lo2, d_flags_lo1, &lo_prime_cnts[8 ], &lo_prime_cnts[9 ], 23);
    number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_lo2, d_primes_lo1, d_flags_lo1, &lo_prime_cnts[9 ], &lo_prime_cnts[10], 29);
    number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_lo1, d_primes_lo2, d_flags_lo1, &lo_prime_cnts[10], &lo_prime_cnts[11], 31);
    cudaMemset(d_flags_lo1, 0, delta*sizeof(bool));
    number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_lo2, d_primes_lo1, d_flags_lo1, &lo_prime_cnts[11], &lo_prime_cnts[12], 37);
    auto end = thrust::copy_if(lo1_ptr, lo1_ptr + delta-1, lo_flags1_ptr, lo1_ptr, cuda::std::identity());
    thrust::sort(lo1_ptr, end);

/*    number_is_composite_kernel_count_2 <<<(delta/2+255)/256, 256>>>(d_primes_lo, d_flags_lo, delta);
    number_is_composite_kernel_count_3 <<<(delta/2+255)/256, 256>>>(d_primes_lo, d_flags_lo, delta);
    number_is_composite_kernel_count_5 <<<(delta/2+255)/256, 256>>>(d_primes_lo, d_flags_lo, delta);
    number_is_composite_kernel_count_7 <<<(delta/2+255)/256, 256>>>(d_primes_lo, d_flags_lo, delta);
    number_is_composite_kernel_count_11<<<(delta/2+255)/256, 256>>>(d_primes_lo, d_flags_lo, delta);
    number_is_composite_kernel_count_13<<<(delta/2+255)/256, 256>>>(d_primes_lo, d_flags_lo, delta);
    number_is_composite_kernel_count_17<<<(delta/2+255)/256, 256>>>(d_primes_lo, d_flags_lo, delta);
    number_is_composite_kernel_count_19<<<(delta/2+255)/256, 256>>>(d_primes_lo, d_flags_lo, delta);
    number_is_composite_kernel_count_23<<<(delta/2+255)/256, 256>>>(d_primes_lo, d_flags_lo, delta);
    number_is_composite_kernel_count_29<<<(delta/2+255)/256, 256>>>(d_primes_lo, d_flags_lo, delta);
    number_is_composite_kernel_count_31<<<(delta/2+255)/256, 256>>>(d_primes_lo, d_flags_lo, delta);
    cudaMemset(lo_prime_cnt1, 0, sizeof(u32));
    number_is_composite_kernel_count_37<<<(delta/2+255)/256, 256>>>(d_primes_lo1, d_primes_lo2, d_flags_lo, d_flags_lo2, delta, lo_prime_cnt1);*/
    //print_arrays<<<1,1>>>(delta, d_flags_lo2, d_primes_lo2);
    //std::cout << "done lo sort"<< std::endl;
/*    print_array<<<1,1>>>(&lo_prime_cnts[12], d_primes_lo1);
    cudaDeviceSynchronize();
    std::cout << "done print"<< std::endl;
    return 1;*/
    u64 i=6;
    for (; i < to_check; i+=delta) {
        u64 starting_number = i-delta;
        if (i< delta)
            starting_number = 0;
//        std::cout << "starting number: " << starting_number << std::endl;
/*        cudaMemset(d_flags_hi1, 0, delta*2*sizeof(bool));
        cudaMemset(d_flags_hi2, 0, delta*2*sizeof(bool));
        number_is_maybe_prime_kernel<<<(delta+255)/256, 256>>>(starting_number, d_primes_hi1, d_flags_hi1, delta*2);
        number_is_composite_kernel_2 <<<(delta+255)/256, 256>>>(d_primes_hi1, d_flags_hi1, delta*2);
        number_is_composite_kernel_3 <<<(delta+255)/256, 256>>>(d_primes_hi1, d_flags_hi1, delta*2);
        number_is_composite_kernel_5 <<<(delta+255)/256, 256>>>(d_primes_hi1, d_flags_hi1, delta*2);
        number_is_composite_kernel_7 <<<(delta+255)/256, 256>>>(d_primes_hi1, d_flags_hi1, delta*2);
        number_is_composite_kernel_11<<<(delta+255)/256, 256>>>(d_primes_hi1, d_flags_hi1, delta*2);
        number_is_composite_kernel_13<<<(delta+255)/256, 256>>>(d_primes_hi1, d_flags_hi1, delta*2);
        number_is_composite_kernel_17<<<(delta+255)/256, 256>>>(d_primes_hi1, d_flags_hi1, delta*2);
        number_is_composite_kernel_19<<<(delta+255)/256, 256>>>(d_primes_hi1, d_flags_hi1, delta*2);
        number_is_composite_kernel_23<<<(delta+255)/256, 256>>>(d_primes_hi1, d_flags_hi1, delta*2);
        number_is_composite_kernel_29<<<(delta+255)/256, 256>>>(d_primes_hi1, d_flags_hi1, delta*2);
        number_is_composite_kernel_31<<<(delta+255)/256, 256>>>(d_primes_hi1, d_flags_hi1, delta*2);
        cudaMemset(hi_prime_cnts, 0, 13*sizeof(u32));
        number_is_composite_kernel_count_37<<<(delta+255)/256, 256>>>(d_primes_hi1, d_primes_hi2, d_flags_hi1, d_flags_hi2, delta*2, &hi_prime_cnts[0]);
        //print_arrays<<<1,1>>>(delta, d_flags_hi_final, d_primes_hi_final);
        auto end = thrust::copy_if(hi1_ptr, hi1_ptr + delta*2-1, hi_flags1_ptr, hi1_ptr, cuda::std::identity());
        thrust::sort(hi1_ptr, end);*/
//        std::cout << "done hi sort"<< std::endl;

        cudaMemset(hi_prime_cnts, 0, 13*sizeof(u32));
        cudaMemset(d_flags_hi1, 0, delta*sizeof(bool));
        number_is_maybe_prime_kernel_count<<<(delta+255)/256, 256>>>(starting_number, d_primes_hi1, d_flags_hi1, delta*2, &hi_prime_cnts[0]);
        number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_hi1, d_primes_hi2, d_flags_hi1, &hi_prime_cnts[0 ], &hi_prime_cnts[1 ], 2);
        number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_hi2, d_primes_hi1, d_flags_hi1, &hi_prime_cnts[1 ], &hi_prime_cnts[2 ], 3);
        number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_hi1, d_primes_hi2, d_flags_hi1, &hi_prime_cnts[2 ], &hi_prime_cnts[3 ], 5);
        number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_hi2, d_primes_hi1, d_flags_hi1, &hi_prime_cnts[3 ], &hi_prime_cnts[4 ], 7);
        number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_hi1, d_primes_hi2, d_flags_hi1, &hi_prime_cnts[4 ], &hi_prime_cnts[5 ], 11);
        number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_hi2, d_primes_hi1, d_flags_hi1, &hi_prime_cnts[5 ], &hi_prime_cnts[6 ], 13);
        number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_hi1, d_primes_hi2, d_flags_hi1, &hi_prime_cnts[6 ], &hi_prime_cnts[7 ], 17);
        number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_hi2, d_primes_hi1, d_flags_hi1, &hi_prime_cnts[7 ], &hi_prime_cnts[8 ], 19);
        number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_hi1, d_primes_hi2, d_flags_hi1, &hi_prime_cnts[8 ], &hi_prime_cnts[9 ], 23);
        number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_hi2, d_primes_hi1, d_flags_hi1, &hi_prime_cnts[9 ], &hi_prime_cnts[10], 29);
        number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_hi1, d_primes_hi2, d_flags_hi1, &hi_prime_cnts[10], &hi_prime_cnts[11], 31);
        cudaMemset(d_flags_hi1, 0, delta*sizeof(bool));
        number_is_composite_kernel_count_wrap<<<1,1>>>(d_primes_hi2, d_primes_hi1, d_flags_hi1, &hi_prime_cnts[11], &hi_prime_cnts[12], 37);
        auto end = thrust::copy_if(hi1_ptr, hi1_ptr + delta-1, hi_flags1_ptr, hi1_ptr, cuda::std::identity());
        thrust::sort(hi1_ptr, end);
        cudaDeviceSynchronize();
        goldbach_kernel<<<(delta/2+31)/32,32>>>(i, i+delta, d_primes_lo1, d_primes_hi1, &lo_prime_cnts[12], &hi_prime_cnts[12]);
//        goldbach_kernel<<<,1>>>(i, i+delta, d_primes_lo, d_primes_hi, lo_prime_cnt, hi_prime_cnt);

        int percent = (i * 100) / to_check;
        int prev_percent = ((i - delta) * 100) / to_check;

        if (percent > prev_percent || percent == 0) {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - begin;

            double progress = static_cast<double>(i+delta) / to_check;
            double total_estimated_seconds = elapsed.count() / progress;
            double remaining_seconds = total_estimated_seconds - elapsed.count();

            std::string remaining_time_str = format_duration(std::chrono::seconds(static_cast<int>(remaining_seconds)));

            std::cout << "\033[s " << percent << "% done | Remaining: " << remaining_time_str << " \033[u" << std::flush;
        }
    }
    std::cout << "\033[s100% done "<<std::endl;
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point tend = std::chrono::steady_clock::now();
    float s = std::chrono::duration_cast<std::chrono::milliseconds> (tend - begin).count()/1000.0;
    now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "\nEnded calculation: " << ctime(&now) << std::endl;
    
    std::cout << "Elapsed time: " << s << "sec" << std::endl;
    std::cout << "Checked: "<<to_check <<" (" << to_check/s << " per sec)" << std::endl;

    return 0;
}
/*#include <chrono>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
int main() {
    cudaSetDevice(0);  // or whatever valid device you want
    time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "Starting calculation: " << ctime(&now) << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    u64 *d_primes_lo;
    u64 *d_primes_hi;
    bool *d_flags_lo;
    bool *d_flags_hi;
    u32 *hi_prime_cnt;
    u32 *lo_prime_cnt;
    cudaMalloc(&d_primes_lo, delta * sizeof(u64));
    cudaMalloc(&d_flags_lo, delta * sizeof(bool));
    cudaMalloc(&d_primes_hi, delta * 2 * sizeof(u64));
    cudaMalloc(&d_flags_hi, delta * 2 * sizeof(bool));
    cudaMemset(d_flags_lo, 0, delta*sizeof(bool));

    cudaMalloc(&hi_prime_cnt, sizeof(u32));
    cudaMalloc(&lo_prime_cnt, sizeof(u32));
    cudaMemset(lo_prime_cnt, 0, sizeof(u32));
    miller_rabin_64_kernel<<<blocks1, threads_per_block>>>(0, d_primes_lo, d_flags_lo ,lo_prime_cnt, delta, 1);
    // Allocate temp storage
    // Wrap the raw pointer in a thrust device_ptr
    thrust::device_ptr<u64> lo_ptr = thrust::device_pointer_cast(d_primes_lo);
    thrust::device_ptr<u64> hi_ptr = thrust::device_pointer_cast(d_primes_hi);
    thrust::device_ptr<bool> lo_flags_ptr = thrust::device_pointer_cast(d_flags_lo);
    thrust::device_ptr<bool> hi_flags_ptr = thrust::device_pointer_cast(d_flags_hi);

    // Sort in-place

//    cudaMemcpy(&host_lo_prime_cnt, lo_prime_cnt, sizeof(u32), cudaMemcpyDeviceToHost);
//    std::cout << "starting lo sort"<< std::endl;
    auto end = thrust::copy_if(lo_ptr, lo_ptr + delta-1, lo_flags_ptr, lo_ptr, cuda::std::identity());
    thrust::sort(lo_ptr, end);
//    std::cout << "done lo sort"<< std::endl;
//    print_primes<<<1,1>>>(lo_prime_cnt, d_primes_lo);
//    std::cout << "done print"<< std::endl;
//    cudaDeviceSynchronize();

    u64 i=6;
    for (; i < to_check; i+=delta) {
        u64 starting_number = i-delta;
        if (i< delta)
            starting_number = 0;
//        std::cout << "starting number: " << starting_number << std::endl;
        cudaMemset(hi_prime_cnt, 0, sizeof(u32));
        cudaMemset(d_flags_hi, 0, delta*2*sizeof(bool));
        miller_rabin_64_kernel<<<blocks2, threads_per_block>>>(starting_number, d_primes_hi, d_flags_hi, hi_prime_cnt, delta*2,1);

//        std::cout << "starting hi sort" << std::endl;
        end = thrust::copy_if(hi_ptr, hi_ptr + delta*2-1, hi_flags_ptr, hi_ptr, cuda::std::identity());
        thrust::sort(hi_ptr, end);
//        std::cout << "done hi sort"<< std::endl;

        goldbach_kernel<<<(delta/2+31)/32,32>>>(i, i+delta, d_primes_lo, d_primes_hi, lo_prime_cnt, hi_prime_cnt);
//        goldbach_kernel<<<,1>>>(i, i+delta, d_primes_lo, d_primes_hi, lo_prime_cnt, hi_prime_cnt);

    }
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point tend = std::chrono::steady_clock::now();
    float s = std::chrono::duration_cast<std::chrono::milliseconds> (tend - begin).count()/1000.0;
    now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << "Ended calculation: " << ctime(&now) << std::endl;
    
    std::cout << "Elapsed time: " << s << "sec" << std::endl;
    std::cout << "Checked: "<<to_check <<" (" << to_check/s << " per sec)" << std::endl;

    return 0;
}*/
