#ifndef HTTP_HPP
#define HTTP_HPP

#include <cstring>
#include <unordered_map>
#include "common/json.hpp"
#include "socket.hpp"

using json = nlohmann::json;

namespace HTTP {
    enum class HttpMethod {
        METHOD_GET = 0,
        METHOD_POST = 1,
        METHOD_PUT = 2,
        METHOD_DELETE = 3,
        METHOD_UNKNOWN = 4
    };

    class HttpRequest {
    public:
        std::string path;
        std::unordered_map<std::string, std::string> headers;
        std::string body;
        json parsedJson;
        HttpMethod method;
        std::string getMethod();
    };

    class HttpParser {
    public:
        static HttpRequest parseRequest(const std::string& request);
    private:
        static HttpMethod parseMethod(const std::string& method);
    };

    struct Route {
        std::string path;
        HttpMethod method;
        std::function<void(Socket&, HttpRequest&)> handler;
    };

    class Router {
    public:
        static void routeRequest(Socket& client_socket, HttpRequest& request, std::vector<Route>& routes);
    private:
        static void notFoundHandler(Socket& request);
    };
}


#endif // HTTP_HPP