#ifndef worker_hpp
#define worker_hpp

class Worker {
private:
    static void listenClient(int clientSocket);
public:
    static void serve(int port);
};

#endif
