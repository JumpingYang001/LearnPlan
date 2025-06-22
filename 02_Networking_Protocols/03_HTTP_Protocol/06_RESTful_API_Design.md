# RESTful API Design with HTTP

## Overview
REST principles, resource identification, HTTP methods for CRUD, status codes, hypermedia, content types, versioning, authentication, and rate limiting.

## C/C++ Example: RESTful API Client (libcurl)
```c
// Simple RESTful GET using libcurl
#include <curl/curl.h>
int main() {
    CURL *curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "https://api.example.com/resource");
        curl_easy_perform(curl);
        curl_easy_cleanup(curl);
    }
    return 0;
}
```
