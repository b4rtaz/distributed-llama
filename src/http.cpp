#include "http.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#endif

#include "nn/nn-network.hpp"

class HttpRequestException : public std::runtime_error {
public:
    explicit HttpRequestException(const std::string &message) : std::runtime_error(message) {}
};

static HttpMethod parseMethod(const std::string &method) {
    if (method == "GET") return HttpMethod::METHOD_GET;
    if (method == "POST") return HttpMethod::METHOD_POST;
    if (method == "PUT") return HttpMethod::METHOD_PUT;
    if (method == "DELETE") return HttpMethod::METHOD_DELETE;
    if (method == "OPTIONS") return HttpMethod::METHOD_OPTIONS;
    return HttpMethod::METHOD_UNKNOWN;
}

static std::vector<char> readHttpRequest(int socket) {
    std::string httpRequest;
    char buffer[1024 * 64];
    ssize_t bytesRead;

    // First, read all headers.
    std::string headerData;
    std::string extraReadPastHeader;
    for (;;) {
        bytesRead = recv(socket, buffer, sizeof(buffer) - 1, 0);
        if (bytesRead <= 0)
            throw HttpRequestException("Error while reading headers from socket");
        buffer[bytesRead] = '\0';
        headerData.append(buffer);

        const size_t endRnRn = headerData.find("\r\n\r\n");
        if (endRnRn != std::string::npos) {
            if (endRnRn < headerData.size() - 4)
                extraReadPastHeader = headerData.substr(endRnRn + 4);
            break;
        }
        const size_t endNN = headerData.find("\n\n");
        if (endNN != std::string::npos) {
            if (endNN < headerData.size() - 2)
                extraReadPastHeader = headerData.substr(endNN + 2);
            break;
        }
    }

    httpRequest.append(headerData);

    // Find the Content-Length header to determine the body length.
    std::istringstream headerStream(headerData);
    std::string line;
    ssize_t contentLength = 0;
    while (std::getline(headerStream, line) && line != "\r" && line != "\n") {
        size_t pos = line.find(':');
        if (pos == std::string::npos)
            continue;

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 2);
        if (key == "Content-Length") {
            try {
                contentLength = std::stoi(value);
            } catch (const std::invalid_argument&) {
                throw HttpRequestException("Bad Content-Length header - not a number");
            }
            break;
        }
    }

    // Read any body bytes which were not received with the headers.
    if (contentLength > 0) {
        if (extraReadPastHeader.size() > static_cast<size_t>(contentLength))
            throw HttpRequestException("Received more body data than Content-Length header said");
        contentLength -= extraReadPastHeader.size();

        std::vector<char> body(contentLength);
        ssize_t totalRead = 0;
        while (totalRead < contentLength) {
            bytesRead = recv(socket, body.data() + totalRead, contentLength - totalRead, 0);
            if (bytesRead <= 0)
                throw HttpRequestException("Error while reading body from socket");
            totalRead += bytesRead;
        }
        if (!body.empty())
            httpRequest.append(body.data(), contentLength);
    }

    return std::vector<char>(httpRequest.begin(), httpRequest.end());
}

static HttpRequest parseRequest(int socket) {
    HttpRequest request(socket);
    std::vector<char> rawRequest = readHttpRequest(socket);
    std::string data(rawRequest.begin(), rawRequest.end());

    std::istringstream input(data);
    std::string line;
    std::getline(input, line);

    std::istringstream requestLine(line);
    std::string method;
    requestLine >> method >> request.path;
    request.method = parseMethod(method);

    while (std::getline(input, line) && line != "\r") {
        size_t pos = line.find(':');
        if (pos == std::string::npos)
            continue;

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 2);
        value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
            return std::isspace(c) || !std::isprint(c);
        }), value.end());
        request.headers[key] = value;
    }

    std::getline(input, request.body, '\0');
    if (!request.body.empty())
        request.parsedJson = nlohmann::json::parse(request.body, nullptr, false);
    return request;
}

static void resolveRoute(HttpRequest &request, const std::vector<Route> &routes) {
    if (request.method == HttpMethod::METHOD_OPTIONS) {
        request.writeCors();
        return;
    }
    for (const Route &route : routes) {
        if (request.method == route.method && request.path == route.path) {
            route.handler(request);
            return;
        }
    }
    request.writeNotFound();
}

HttpRequest::HttpRequest(int socket) : socket(socket), method(HttpMethod::METHOD_UNKNOWN) {}

std::string HttpRequest::getMethod() const {
    if (method == HttpMethod::METHOD_GET) return "GET";
    if (method == HttpMethod::METHOD_POST) return "POST";
    if (method == HttpMethod::METHOD_PUT) return "PUT";
    if (method == HttpMethod::METHOD_DELETE) return "DELETE";
    if (method == HttpMethod::METHOD_OPTIONS) return "OPTIONS";
    return "UNKNOWN";
}

void HttpRequest::writeCors() {
    std::ostringstream buffer;
    buffer << "HTTP/1.1 204 No Content\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Methods: GET, POST, PUT, DELETE\r\n"
        << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        << "Connection: close\r\n"
        << "\r\n";
    std::string data = buffer.str();
    writeSocket(socket, data.c_str(), data.size());
}

void HttpRequest::writeNotFound() {
    std::ostringstream buffer;
    buffer << "HTTP/1.1 404 Not Found\r\n"
        << "Connection: close\r\n"
        << "Content-Length: 9\r\n"
        << "\r\n"
        << "Not Found";
    std::string data = buffer.str();
    writeSocket(socket, data.c_str(), data.size());
}

static const char *httpStatusText(int status) {
    if (status == 200) return "OK";
    if (status == 400) return "Bad Request";
    if (status == 404) return "Not Found";
    if (status == 405) return "Method Not Allowed";
    if (status == 422) return "Unprocessable Entity";
    if (status == 500) return "Internal Server Error";
    return "Error";
}

void HttpRequest::writeJson(const std::string &json, int status) {
    std::ostringstream buffer;
    buffer << "HTTP/1.1 " << status << " " << httpStatusText(status) << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Content-Type: application/json; charset=utf-8\r\n"
        << "Connection: close\r\n"
        << "Content-Length: " << json.length() << "\r\n\r\n" << json;
    std::string data = buffer.str();
    writeSocket(socket, data.c_str(), data.size());
}

void HttpRequest::writeStreamStartChunk() {
    std::ostringstream buffer;
    buffer << "HTTP/1.1 200 OK\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Content-Type: text/event-stream; charset=utf-8\r\n"
        << "Connection: close\r\n"
        << "Transfer-Encoding: chunked\r\n\r\n";
    std::string data = buffer.str();
    writeSocket(socket, data.c_str(), data.size());
}

void HttpRequest::writeStreamChunk(const std::string &data) {
    std::ostringstream buffer;
    buffer << std::hex << data.size() << "\r\n" << data << "\r\n";
    std::string chunk = buffer.str();
    writeSocket(socket, chunk.c_str(), chunk.size());
}

void HttpRequest::writeStreamEndChunk() {
    const char *endChunk = "0000\r\n\r\n";
    writeSocket(socket, endChunk, strlen(endChunk));
}

void runHttpServer(const char *host, int port, const std::vector<Route> &routes) {
    NnSocket serverSocket(createServerSocket(host, port));

    while (true) {
        try {
            NnSocket clientSocket(acceptSocket(serverSocket.fd));
            HttpRequest request = parseRequest(clientSocket.fd);
            printf("🔷 %s %s\n", request.getMethod().c_str(), request.path.c_str());
            resolveRoute(request, routes);
        } catch (const NnTransferSocketException &e) {
            printf("Socket error: %d %s\n", e.code, e.what());
        } catch (const HttpRequestException &e) {
            printf("HTTP request error: %s\n", e.what());
        }
    }
}
