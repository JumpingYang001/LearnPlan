# Project: RESTful API Client

## Description
Build a client library for interacting with RESTful APIs. Support authentication and content negotiation.

## C/C++ Example: RESTful API GET with libcurl
```c
#include <curl/curl.h>
int main() {
    CURL *curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "https://api.example.com/data");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_slist_append(NULL, "Accept: application/json"));
        curl_easy_perform(curl);
        curl_easy_cleanup(curl);
    }
    return 0;
}
```
