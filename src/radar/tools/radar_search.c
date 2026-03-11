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
    fprintf(stderr, "Usage: %s <query>\n", argv[0]);
    return 1;
  }

  CURL *curl_handle;
  CURLcode res;
  struct MemoryStruct chunk;

  chunk.memory = malloc(1);
  chunk.size = 0;

  curl_global_init(CURL_GLOBAL_ALL);
  curl_handle = curl_handle = curl_easy_init();

  // Construct DuckDuckGo HTML search URL
  char url[1024];
  char *encoded_query = curl_easy_escape(curl_handle, argv[1], 0);
  snprintf(url, sizeof(url), "https://html.duckduckgo.com/html/?q=%s", encoded_query);
  curl_free(encoded_query);

  curl_easy_setopt(curl_handle, CURLOPT_URL, url);
  curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
  curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void *)&chunk);
  curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "Mozilla/5.0 (compatible; RadarBot/1.0)");
  curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L);

  res = curl_easy_perform(curl_handle);

  if(res != CURLE_OK) {
    fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
  } else {
    // Simple parser for links in DuckDuckGo HTML
    // Looking for result__a href="..."
    char *ptr = chunk.memory;
    int count = 0;
    while ((ptr = strstr(ptr, "result__a\" href=\"")) != NULL && count < 5) {
      ptr += 17;
      char *end = strstr(ptr, "\"");
      if (end) {
        *end = '\0';
        // Some DDG links are redirects, we try to extract the real URL if it's there
        char *uddg = strstr(ptr, "uddg=");
        if (uddg) {
            uddg += 5;
            // Basic URL decoding for the uddg param would be better, 
            // but for now we'll just print it.
            printf("%s\n", uddg);
        } else {
            printf("%s\n", ptr);
        }
        *end = '\"';
        ptr = end + 1;
        count++;
      }
    }
  }

  curl_easy_cleanup(curl_handle);
  free(chunk.memory);
  curl_global_cleanup();

  return 0;
}
