#include "Server.h"

using namespace std;

Server::Server() {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        cerr << "Failed to initialize Winsock." << endl;
        exit(EXIT_FAILURE);
    }

    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == INVALID_SOCKET) {
        cerr << "Failed to create server socket." << endl;
        WSACleanup();
        exit(EXIT_FAILURE);
    }

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(PORT);

    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        cerr << "Bind failed." << endl;
        closesocket(serverSocket);
        WSACleanup();
        exit(EXIT_FAILURE);
    }

    if (listen(serverSocket, SOMAXCONN) == SOCKET_ERROR) {
        cerr << "Listen failed." << endl;
        closesocket(serverSocket);
        WSACleanup();
        exit(EXIT_FAILURE);
    }

    cout << "Server listening on port " << PORT << endl;
}

Server::~Server() {
    closesocket(serverSocket);
    WSACleanup();
}

void Server::acceptConnections() {
    while (true) {
        SOCKET clientSocket = accept(serverSocket, nullptr, nullptr);
        if (clientSocket == INVALID_SOCKET) {
            cerr << "Failed to create socket" << endl;
            closesocket(serverSocket);
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        handleClient(clientSocket);
    }
}

void Server::handleClient(SOCKET clientSocket) {

    // Send a "Hello" message back to the client
    string helloMessage = "Hello from the server!";
    send(clientSocket, helloMessage.c_str(), helloMessage.length(), 0);

    closesocket(clientSocket);
}
