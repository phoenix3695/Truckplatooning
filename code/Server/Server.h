// Server.h
#pragma once

#include <iostream>
#include <string>
#include <WinSock2.h>

#pragma comment(lib, "ws2_32.lib")

#define PORT 8080
#define BUFFER_SIZE 1024

class Server {
public:
    Server();
    ~Server();

    void acceptConnections();

private:
    SOCKET serverSocket;
    sockaddr_in serverAddr;

    void handleClient(SOCKET clientSocket);
};
