#pragma once

#include <iostream>
#include <string>
#include <WinSock2.h>

#pragma comment(lib, "ws2_32.lib")

#define PORT 8080
#define BUFFER_SIZE 1024

class Client {
public:
    Client();
    ~Client();

    void communicate();

private:
    SOCKET clientSocket;
    sockaddr_in serverAddr;
};
