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
    int line_limit = 3; // Default limit
    int is_sitrep = 0;
    
    while (fgets(buffer, MAX_BUFFER, stdin) != NULL && line_count < 30) {
        
        char *p = buffer;
        while (*p == ' ' || *p == '\t' || *p == '-') p++;
        
        if (strncmp(p, "Question:", 9) == 0) {
            printf("🎯 %s\n", p + 10);
            strncpy(query, p + 10, sizeof(query) - 1);
            query[strcspn(query, "\n")] = 0;
            to_lowercase(query);
            
            char *token = strtok(query, " ?.,");
            while (token != NULL && word_count < 10) {
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
                strstr(p, "function()") == NULL &&
                strstr(p, "font-family") == NULL &&
                strstr(p, "LOCAL EXTRACTIVE") == NULL &&
                strstr(p, "found no direct hits") == NULL &&
                strncmp(p, "Autonomous Research", 19) != 0 &&
                strncmp(p, "Target:", 7) != 0 &&
                strlen(p) > 10) {
                
                if (strncmp(p, "Title:", 6) == 0 || strncmp(p, "Source:", 7) == 0) {
                    // Only update title if it's a real title or we don't have one yet
                    if (p[0] == 'T' || strlen(current_title) == 0) {
                        int offset = (p[0] == 'T') ? 7 : 8;
                        strncpy(current_title, p + offset, MAX_BUFFER);
                        current_title[strcspn(current_title, "\n")] = 0;
                        
                        char *dash = strstr(current_title, " ---");
                        if (dash != NULL) *dash = '\0';

                        if (strstr(current_title, "SITREP") != NULL) {
                            line_limit = 15;
                            is_sitrep = 1;
                        } else {
                            line_limit = 3;
                            is_sitrep = 0;
                        }
                    }
                    // Reset content lines for the document, but don't clear the 'sitrep' status
                    content_lines = 0;
                    current_title_printed = 0;
                } else {
                    if (content_lines < line_limit) {
                        int has_digit = 0;
                        for (int i = 0; p[i]; i++) {
                            if (isdigit(p[i])) {
                                has_digit = 1;
                                break;
                            }
                        }

                        if (strlen(p) < 40 && (p[0] == '#' || p[0] == '*' || p[0] == '-') && !has_digit) {
                            continue;
                        }

                        if (strlen(p) < 60 && !has_digit) {
                            continue;
                        }

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
                    }
                }
            }
        }
    }
    
    if (total_content_lines == 0) {
        printf("\n  - No specific qualitative insights extracted from local corpus. Check sources below.\n");
    }
    
    return 0;
}
