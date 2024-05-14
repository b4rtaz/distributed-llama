#include "http.hpp"
#include "common/json.hpp"
#include <sstream>
#include "socket.hpp"
using json = nlohmann::json;

namespace HTTP {
    std::string HttpRequest::getMethod(){
        switch(method) {
            case HttpMethod::METHOD_GET:
                return "GET";
            case HttpMethod::METHOD_POST:
                return "POST";
            case HttpMethod::METHOD_PUT:
                return "PUT";
            case HttpMethod::METHOD_DELETE:
                return "DELETE";
            case HttpMethod::METHOD_UNKNOWN:
            default:
                return "UNKNOWN";
        }
    }

    HttpRequest HttpParser::parseRequest(const std::string& request) {
        HttpRequest httpRequest;

        // Split request into lines
        std::istringstream iss(request);
        std::string line;
        std::getline(iss, line);

        // Parse request line
        std::istringstream lineStream(line);
        std::string methodStr, path;
        lineStream >> methodStr >> path;
        httpRequest.method = parseMethod(methodStr);
        httpRequest.path = path;

        // Parse headers
        while (std::getline(iss, line) && line != "\r") {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 2); // Skip ': ' after key
                // Trim whitespace and non-printable characters from header value
                value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
                    return std::isspace(c) || !std::isprint(c);
                }), value.end());
                httpRequest.headers[key] = value;
            }
        }

        // Parse body
        std::getline(iss, httpRequest.body, '\0');

        // Parse JSON if Content-Type is application/json
        httpRequest.parsedJson = json::object();
        if(httpRequest.headers.find("Content-Type") != httpRequest.headers.end()){
            if(httpRequest.headers["Content-Type"] == "application/json"){
                httpRequest.parsedJson = json::parse(httpRequest.body);
            }
        }

        return httpRequest;
    }

    HttpMethod HttpParser::parseMethod(const std::string& method) {
        if (method == "GET") {
            return HttpMethod::METHOD_GET;
        } else if (method == "POST") {
            return HttpMethod::METHOD_POST;
        } else if (method == "PUT") {
            return HttpMethod::METHOD_PUT;
        } else if (method == "DELETE") {
            return HttpMethod::METHOD_DELETE;
        } else {
            return HttpMethod::METHOD_UNKNOWN;
        }
    }

    void Router::routeRequest(Socket& client_socket, HttpRequest& request, std::vector<Route>& routes) {
        for (const auto& route : routes) {
            if (request.method == route.method && request.path == route.path) {
                route.handler(client_socket, request);
                return;
            }
        }
        notFoundHandler(client_socket);
    }

    void Router::notFoundHandler(Socket& client_socket) {
        std::string header = "HTTP/1.1 404 Not Found\r\n";
        client_socket.write(header.c_str(), header.length());
    }
}