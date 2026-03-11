#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>

struct MemoryStruct {
  char *memory;
  size_t size;
};

static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp) {
  size_t realsize = size * nmemb;
  struct MemoryStruct *mem = (struct MemoryStruct *)userp;

  char *ptr = realloc(mem->memory, mem->size + realsize + 1);
  if(!ptr) return 0;

  mem->memory = ptr;
  memcpy(&(mem->memory[mem->size]), contents, realsize);
  mem->size += realsize;
  mem->memory[mem->size] = 0;

  return realsize;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <url>\n", argv[0]);
    return 1;
  }

  CURL *curl_handle;
  CURLcode res;
  struct MemoryStruct chunk;

  chunk.memory = malloc(1);
  chunk.size = 0;

  curl_global_init(CURL_GLOBAL_ALL);
  curl_handle = curl_easy_init();

  curl_easy_setopt(curl_handle, CURLOPT_URL, argv[1]);
  curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
  curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void *)&chunk);
  curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36");
  curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl_handle, CURLOPT_MAXREDIRS, 5L);
  curl_easy_setopt(curl_handle, CURLOPT_TIMEOUT, 15L);
  curl_easy_setopt(curl_handle, CURLOPT_SSL_VERIFYPEER, 0L); 
  curl_easy_setopt(curl_handle, CURLOPT_SSL_VERIFYHOST, 0L);

  res = curl_easy_perform(curl_handle);

  if(res != CURLE_OK) {
    fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
  } else {
    // Determine if we should output raw binary (for PDFs) or extracted text (for HTML)
    int is_pdf = 0;
    int len = strlen(argv[1]);
    if (len > 4 && strcasecmp(argv[1] + len - 4, ".pdf") == 0) {
        is_pdf = 1;
    }

    if (is_pdf) {
        // Output raw binary data for PDF processing in Python
        fwrite(chunk.memory, 1, chunk.size, stdout);
    } else {
        // Basic text extraction for HTML
        int in_tag = 0;
        int in_script = 0;
        int consecutive_spaces = 0;
        
        for (size_t i = 0; i < chunk.size; i++) {
            char c = chunk.memory[i];
            
            if (i < chunk.size - 7 && strncasecmp(&chunk.memory[i], "<script", 7) == 0) in_script = 1;
            if (i < chunk.size - 6 && strncasecmp(&chunk.memory[i], "<style", 6) == 0) in_script = 1;
            if (in_script && i > 8 && strncasecmp(&chunk.memory[i-9], "</script>", 9) == 0) in_script = 0;
            if (in_script && i > 7 && strncasecmp(&chunk.memory[i-8], "</style>", 8) == 0) in_script = 0;

            if (c == '<') {
                in_tag = 1;
            } else if (c == '>') {
                in_tag = 0;
                if (!consecutive_spaces) {
                    putchar('\n');
                    consecutive_spaces = 1;
                }
            } else if (!in_tag && !in_script) {
                if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
                    if (!consecutive_spaces) {
                        putchar(' ');
                        consecutive_spaces = 1;
                    }
                } else {
                    putchar(c);
                    consecutive_spaces = 0;
                }
            }
        }
        printf("\n");
    }
  }

  curl_easy_cleanup(curl_handle);
  free(chunk.memory);
  curl_global_cleanup();

  return 0;
}
