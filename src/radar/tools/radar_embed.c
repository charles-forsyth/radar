#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define DIMENSIONS 3072
#define MAX_TOKEN_LEN 256

// Simple hashing function (djb2)
unsigned long hash(const char *str) {
    unsigned long h = 5381;
    int c;
    while ((c = *str++))
        h = ((h << 5) + h) + c; 
    return h;
}

// Pseudorandom number generator for consistent projections
float get_random_weight(unsigned long seed, int dim) {
    // Deterministic "random" weight based on seed and dimension
    unsigned long h = seed ^ (dim * 2654435761u);
    h = (h ^ (h >> 16)) * 0x85ebca6bu;
    h = (h ^ (h >> 13)) * 0xc2b2ae35u;
    h = h ^ (h >> 16);
    return (float)(h % 2000 - 1000) / 1000.0f; // Range [-1.0, 1.0]
}

int main(int argc, char *argv[]) {
    float *vec = (float *)calloc(DIMENSIONS, sizeof(float));
    char token[MAX_TOKEN_LEN];
    int token_idx = 0;
    
    int c;
    while ((c = fgetc(stdin)) != EOF) {
        if (isalnum(c)) {
            if (token_idx < MAX_TOKEN_LEN - 1) {
                token[token_idx++] = tolower(c);
            }
        } else {
            if (token_idx > 0) {
                token[token_idx] = '\0';
                unsigned long token_hash = hash(token);
                // Project this token into the dense vector space
                for (int i = 0; i < DIMENSIONS; i++) {
                    vec[i] += get_random_weight(token_hash, i);
                }
                token_idx = 0;
            }
        }
    }
    if (token_idx > 0) {
        token[token_idx] = '\0';
        unsigned long token_hash = hash(token);
        for (int i = 0; i < DIMENSIONS; i++) {
            vec[i] += get_random_weight(token_hash, i);
        }
    }
    
    // L2 Normalize
    float sum_sq = 0.0f;
    for (int i = 0; i < DIMENSIONS; i++) {
        sum_sq += vec[i] * vec[i];
    }
    if (sum_sq > 0.0f) {
        float norm = sqrtf(sum_sq);
        for (int i = 0; i < DIMENSIONS; i++) {
            vec[i] /= norm;
        }
    }
    
    printf("[");
    for (int i = 0; i < DIMENSIONS; i++) {
        printf("%.6f", vec[i]);
        if (i < DIMENSIONS - 1) printf(", ");
    }
    printf("]\n");
    
    free(vec);
    return 0;
}
