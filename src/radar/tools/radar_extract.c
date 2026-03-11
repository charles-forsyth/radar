#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_ENTITIES 100
#define MAX_TOKEN_LEN 256

int main() {
    char token[MAX_TOKEN_LEN];
    char entities[MAX_ENTITIES][MAX_TOKEN_LEN];
    int entity_count = 0;
    int token_idx = 0;
    int in_word = 0;
    int is_capitalized = 0;

    int c;
    while ((c = fgetc(stdin)) != EOF) {
        if (isalnum(c) || c == '-') {
            if (!in_word) {
                in_word = 1;
                is_capitalized = isupper(c);
            }
            if (token_idx < MAX_TOKEN_LEN - 1) {
                token[token_idx++] = c;
            }
        } else {
            if (in_word) {
                token[token_idx] = '\0';
                if (is_capitalized && token_idx > 3 && entity_count < MAX_ENTITIES) {
                    int dup = 0;
                    for (int i = 0; i < entity_count; i++) {
                        if (strcmp(entities[i], token) == 0) {
                            dup = 1;
                            break;
                        }
                    }
                    if (!dup) {
                        strcpy(entities[entity_count++], token);
                    }
                }
                token_idx = 0;
                in_word = 0;
            }
        }
    }
    
    printf("{\n");
    printf("  \"entities\": [\n");
    for (int i = 0; i < entity_count; i++) {
        // Use EntityType: TECH, COMPANY, PERSON, MARKET
        printf("    {\"name\": \"%s\", \"type\": \"TECH\", \"description\": \"Extracted local entity\"}%s\n", 
               entities[i], (i < entity_count - 1) ? "," : "");
    }
    printf("  ],\n");
    printf("  \"connections\": [],\n");
    printf("  \"trends\": []\n");
    printf("}\n");

    return 0;
}
