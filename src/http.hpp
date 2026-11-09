#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"

enum class HttpMethod {
    METHOD_GET = 0,
    METHOD_POST = 1,
    METHOD_PUT = 2,
    METHOD_DELETE = 3,
    METHOD_OPTIONS = 4,
    METHOD_UNKNOWN = 5
};

class HttpRequest {
private:
    int socket;

public:
    std::string path;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    nlohmann::json parsedJson;
    HttpMethod method;

    explicit HttpRequest(int socket);

    std::string getMethod() const;
    void writeCors();
    void writeNotFound();
    void writeJson(const std::string &json, int status = 200);
    void writeStreamStartChunk();
    void writeStreamChunk(const std::string &data);
    void writeStreamEndChunk();
};

struct Route {
    std::string path;
    HttpMethod method;
    std::function<void(HttpRequest&)> handler;
};

void runHttpServer(const char *host, int port, const std::vector<Route> &routes);
