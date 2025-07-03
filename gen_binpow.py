template = '''#include <stdint.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef __int128_t i128;

template<u32 N>
struct C {
    constexpr C() : b(),len(0) {
        u64 base = N;
        for (auto i = 0; i < 64 && base < 1L<<32; ++i){
            b[i] = base;
            base *= base;
            len++;
        }
    }
    u8 len;
    u64 b[5];
};
__device__
u64 binpower(u64 base, u64 e, u64 mod) {
    u64 result = 1;
    base %= mod;
    while (e) {
        if (e & 1)
            result = (i128)result * base % mod;
        base = (i128)base * base % mod;
        e >>= 1;
    }
    return result;
}
__device__
bool check_composite(u64 n, u64 a, u64 d, int s) {
    u64 x = binpower(a, d, n);
    if (x == 1 || x == n - 1)
        return false;
    for (int r = 1; r < s; r++) {
        x = (i128)x * x % n;
        if (x == n - 1)
            return false;
    }
    return true;
}

__device__
u64 binpower_l(u64 base, u64 e, u64 mod) {
    u64 result = 1;
    base %= mod;
    #pragma unroll 64
    for(int i = 0; i<64; i++) {
        result =((e>>i) & 1)?(i128)result*base % mod:result;;
        base = (i128)base * base % mod;
    }
    return result;
}

__device__
bool check_composite_l(u64 n, u64 a, u64 d, int s) {
    u64 x = binpower_l(a, d, n);
    bool result = !(x == 1 || x == n - 1);
    #pragma unroll 64
    for (int r = 1; r < 64; r++) {
        x = (i128)x * x % n;
        result &= (r >= s) | !(x == n - 1);
    }
    return result;
};
'''
print(template)
for base in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
    x = f'__device__\nauto base{base:<2} = C<{base:<2}>();'
    print(x)

for base in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
    basepows = [base]
    while (basepows[-1]*basepows[-1] < (1<<32)) :
        basepows += [basepows[-1]*basepows[-1]]
    lines = '\n'.join([f'    result = ((e>>{i})&1)?(i128)result*{str(b)+" "*(len(str(basepows[-1]))-len(str(b)))}%mod:result;'for i,b in enumerate(basepows)])

        
    x=('__device__\n'
f'u64 binpower{base}_l(u64 e, u64 mod) {{\n'
'''
    i128 result = 1;
''' 
f'{lines}\n    i128 base = {basepows[-1]};\n    int i = {len(basepows)};'
'''
    for (;e>>i && i<64;i++) {
        base = base * base % mod;
        result =((e>>i) & 1)?(i128)result*base % mod:result;
    }
'''
#f'    u64 res2 = binpower({base:<2},e,mod);\n    if (res2!=result) printf("binpower({base:<2},%ld,%ld) = %ld != %ld\\n", e,mod,res2,(u64)result);'
'''
    return result;
}
''')
    print(x)
for base in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
    x=('__device__\n'
f'bool check_composite{base}_l(u64 n, u64 d, int s) {{\n    u64 x = binpower{base}_l(d, n);'
'''
    if (x == 1 || x == n - 1)
        return false;
    for (int r = 1; r < s; r++) {
        x = (i128)x * x % n;
        if (x == n - 1)
            return false;
    }
'''
#f'    bool res2 = check_composite(n,{base:<2},d,s);\n    if (res2!=result) printf("check_composite(%ld,{base:<2},%ld,%d) = %d != %d\\n", n,d,s,res2,result);'
'''
    return true;
};
''')
    print(x)

for base in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
    x=('__device__\n'
f'u64 binpower{base}(u64 e, u64 mod) {{\n    auto lbase = base{base};'
'''
    i128 result = 1;
    int i = 0;
    while (e>>i && i<lbase.len) {
        if ((e>>i) & 1)
            result = result * lbase.b[i] % mod;
        i++; 
    }
    i128 base = lbase.b[lbase.len-1];

    while (e>>i) {
        base = base * base % mod;
        if ((e>>i) & 1)
            result = result * base % mod;
        i++;
    }
    return result;
}
''')
    print(x)
for base in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
    x = ('__device__\n'
f'bool check_composite{base}(u64 n, u64 d, int s) {{\n'
f'    u64 x = binpower{base}(d, n);\n'
#f'    u64 x = binpower({base:<2},d, n);\n'
'''    if (x == 1 || x == n - 1)
        return false;
    for (int r = 1; r < s; r++) {
        x = (i128)x * x % n;
        if (x == n - 1)
            return false;
    }
    return true;
}\n'''
    )
    print(x)
eq = '\n'.join([f"        if (n == {base:<2}) {{flag[idx] = true;return;}}"for base in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]])
for base in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
    x=(f'__global__ void number_is_composite_kernel_{base}(u64* numbers, bool *flag, u32 num_candidates)' '''{
    int tid = threadIdx.x; // [0..12*numbers_per_block-1]
    int idx = blockIdx.x*blockDim.x+tid;

    if (idx >= num_candidates){
        return;
    } else if (!flag[idx]){
        return;
    }

    flag[idx] = false;
    u64 n = numbers[idx];
    u64 d = n - 1;
    int s = __ffsll(d) - 1;
    d >>= s;

''' f'    if ({base:<2} >= n)' '{ \n'
f'{eq}\n'
'''    }
'''
f'    bool is_not_composite = !(check_composite{base}_l(n, d, s));\n'
'''    flag[idx] = is_not_composite;
}
''')
    print(x)

eq2 = '\n'.join([f"        if (n == {base:<2}) {{u32 outidx = atomicAdd(count,1);oflag[outidx] = true;output_numbers[outidx] = {base:<2};return;}}"for base in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]])
for base in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
    x=(f'__global__ void number_is_composite_kernel_count_{base}(u64* numbers,u64* output_numbers, bool *flag, bool *oflag, u32 num_candidates, u32* count)' '''{
    int tid = threadIdx.x; // [0..12*numbers_per_block-1]
    int idx = blockIdx.x*blockDim.x+tid;

    if (idx >= num_candidates){
        return;
    } else if (!flag[idx]){
        return;
    }

    flag[idx] = false;
    u64 n = numbers[idx];
    u64 d = n - 1;
    int s = __ffsll(d) - 1;
    d >>= s;

''' f'    if ({base:<2} >= n)' '{ \n'
f'{eq2}\n'
'''    }
'''
f'    bool is_not_composite = !(check_composite{base}_l(n, d, s));'
'''
    if (is_not_composite){
        u32 outidx = atomicAdd(count,1);
        oflag[outidx] = true;
        output_numbers[outidx] = n;
    }
}
''')
    print(x)
