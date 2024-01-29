#include <iostream>
#include "Server/Server.h"
#include "Client/Client.h"

using namespace std;

int main() {
    int choice;
    cout << "Select mode:\n";
    cout << "1. Leader (Server)\n";
    cout << "2. Follower (Client)\n";
    cout << "Enter choice: ";
    cin >> choice;

    if (choice == 1) {
        Server server;
        server.acceptConnections();
    } else if (choice == 2) {
        Client client;
        client.communicate();
    } else {
        cerr << "Invalid choice. Exiting." << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
