#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BUFFER 8192

int main(int argc, char *argv[]) {
    char buffer[MAX_BUFFER];
    size_t len = 0;
    
    // Extractive summary: just take the first few lines and format them.
    printf("LOCAL EXTRACTIVE BRIEFING:\n");
    printf("===========================\n");
    
    int line_count = 0;
    while (fgets(buffer, MAX_BUFFER, stdin) != NULL && line_count < 10) {
        if (strlen(buffer) > 5) { // Skip empty lines
            printf("- %s", buffer);
            line_count++;
        }
    }
    
    if (line_count == 0) {
        printf("No data provided for briefing.\n");
    }
    
    return 0;
}
