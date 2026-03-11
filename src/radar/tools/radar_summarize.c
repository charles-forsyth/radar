#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_BUFFER 8192

void to_lowercase(char *str) {
    for(int i = 0; str[i]; i++){
        str[i] = tolower((unsigned char)str[i]);
    }
}

int main(int argc, char *argv[]) {
    char buffer[MAX_BUFFER];
    char query[256] = "";
    char words[10][64] = {0};
    int word_count = 0;
    
    int line_count = 0;
    int content_lines = 0;
    int total_content_lines = 0;
    int current_title_printed = 0;
    char current_title[MAX_BUFFER] = "";
    
    while (fgets(buffer, MAX_BUFFER, stdin) != NULL && line_count < 20) {
        
        char *p = buffer;
        while (*p == ' ' || *p == '\t' || *p == '-') p++;
        
        if (strncmp(p, "Question:", 9) == 0) {
            printf("🎯 %s\n", p + 10);
            strncpy(query, p + 10, sizeof(query) - 1);
            query[strcspn(query, "\n")] = 0;
            to_lowercase(query);
            
            // Tokenize query
            char *token = strtok(query, " ?.,");
            while (token != NULL && word_count < 10) {
                // Lowered from >3 to >2 to catch more keywords
                if (strlen(token) > 2) {
                    strncpy(words[word_count++], token, 63);
                }
                token = strtok(NULL, " ?.,");
            }
            continue;
        }
        
        if (strlen(buffer) > 15 && 
            strstr(buffer, "--- Signal:") == NULL && 
            strstr(buffer, "Context: ") == NULL &&
            strstr(buffer, "Link: http") == NULL) {
            
            if (strchr(p, ' ') != NULL && 
                strstr(p, "{") == NULL && 
                strstr(p, "}") == NULL &&
                strstr(p, "var ") == NULL &&
                strstr(p, "window[") == NULL &&
                strstr(p, "font-family") == NULL &&
                strstr(p, "LOCAL EXTRACTIVE") == NULL &&
                strstr(p, "found no direct hits") == NULL &&
                strlen(p) > 10) {
                
                if (strncmp(p, "Title:", 6) == 0) {
                    strncpy(current_title, p + 7, MAX_BUFFER);
                    current_title[strcspn(current_title, "\n")] = 0; // remove newline
                    current_title_printed = 0;
                    content_lines = 0; // Reset for new document
                } else if (strncmp(p, "Source:", 7) == 0) {
                    // Treat Source as a title for Deep Research outputs
                    strncpy(current_title, p + 8, MAX_BUFFER);
                    current_title[strcspn(current_title, "\n")] = 0;
                    
                    // Strip the trailing " ---" if it exists
                    char *dash = strstr(current_title, " ---");
                    if (dash != NULL) {
                        *dash = '\0';
                    }
                    
                    current_title_printed = 0;
                    content_lines = 0;
                } else {
                    char lower_p[MAX_BUFFER];
                    strncpy(lower_p, p, MAX_BUFFER);
                    to_lowercase(lower_p);

                    int relevant = 0;
                    if (word_count > 0) {
                        for (int w = 0; w < word_count; w++) {
                            if (strstr(lower_p, words[w]) != NULL) {
                                relevant = 1;
                                break;
                            }
                        }
                    } else {
                        relevant = 1; 
                    }

                    // We only want to print lines that are actually relevant to the query to avoid noise.
                    // But if it's the very first few lines of a document, they often contain the site header/title
                    // which is good for context.
                    if (relevant || (content_lines < 2 && line_count < 15)) {
                        if (!current_title_printed && strlen(current_title) > 0) {
                            printf("\n📌 %s\n", current_title);
                            current_title_printed = 1;
                        }

                        char display_p[512];
                        if (strlen(p) > 250) {
                            strncpy(display_p, p, 247);
                            display_p[247] = '\0';
                            strcat(display_p, "...");
                        } else {
                            strncpy(display_p, p, sizeof(display_p) - 1);
                            display_p[strcspn(display_p, "\n")] = 0; 
                        }

                        printf("  - %s\n", display_p);
                        content_lines++;
                        total_content_lines++;
                    }                }
            }
        }
    }
    
    if (total_content_lines == 0) {
        printf("\n  - No specific qualitative insights extracted from local corpus. Check sources below.\n");
    }
    
    return 0;
}
