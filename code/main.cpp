#include <iostream>
#include <string>
#include <WinSock2.h>
#include <vector>
#include <thread>
#include <sstream>
#include <mutex>
#include "Truck/LeaderTruck/LeaderTruck.h"
#include "Truck/FollowerTruck/FollowerTruck.h"

#pragma comment(lib, "ws2_32.lib")

#define PORT 8080
#define BUFFER_SIZE 1024

using namespace std;

int logical_clock = 0;
std::mutex clockMutex;
LeaderTruck leader("");
std::mutex leaderMutex;
FollowerTruck follower("");
std::mutex followerMutex;

void handleClient(SOCKET clientSocket) {
    string clientid;
    bool runloop = true;

    string greetingMessage = "Leader:" + leader.getId();
    send(clientSocket, greetingMessage.c_str(), greetingMessage.length(), 0);

    char buffer[BUFFER_SIZE];
    int bytesRead = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
    if (bytesRead > 0) {
        buffer[bytesRead] = '\0';

        string receivedMessage(buffer);
        if (receivedMessage.compare(0, 4, "join") == 0) {
            //cout << receivedMessage.substr(4) << " wants to join" << endl;

            size_t idStart = receivedMessage.find(':', 4);
            size_t idEnd = receivedMessage.find(':', idStart + 1);
            size_t positionStart = receivedMessage.find('[', idEnd + 1);
            size_t positionEnd = receivedMessage.find(']', positionStart + 1);

            if (idStart != string::npos && idEnd != string::npos &&
                positionStart != string::npos && positionEnd != string::npos) {
                string id = receivedMessage.substr(idStart + 1, idEnd - idStart - 1);
                string position = receivedMessage.substr(positionStart + 1, positionEnd - positionStart - 1);
                cout << "Join Request from Follower: " << id << ", Position: " << position << endl;
                clientid = id;
                if (leader.checkIfFollower(clientid) == -1) {
                    leader.addFollower(clientid);
                    cout << "Assigning follower:" << clientid << " to slot " << leader.checkIfFollower(clientid)
                         << endl;
                } else {
                    cerr << "Duplicate request as Follower: " << id << " already exists" << endl;
                    runloop = false;
                    string updateMessage = "ERROR: Duplicate ID";
                    send(clientSocket, updateMessage.c_str(), updateMessage.length(), 0);
                    return;
                }

            }
        }

        //cout << "Received from client: " << receivedMessage << endl;
    }
    int recoupleDuration = 0;
    while (runloop) {
        // Implementation of PID from leadeer side - This was the inititial proposition, changed to pid with each follower.
        //float errors[1] = {1.0f};  // Initialize with appropriate error values
        //float outputs[1] = {0.0f}; // Initialize with appropriate output values
        //float constants[3] = {0.3f, 0.02f, 0.2f};
        //float pidResult = leader.cudaPidComputation(errors, outputs, constants, 1);
        //cout << leader.toString() << endl;
        //Update from server: Update: Some data from the server.
        leader.checkIfFollower(clientid);

        Position expectedPosition = leader.getPosition();
        double size = leader.getSize();
        double gap = leader.getGap();
        int position = leader.checkIfFollower(clientid);
        if (position != -1) {
            for (int e = 0; e <= position; e++) {
                expectedPosition.x = expectedPosition.x + size + gap;
            }
        }
        //logical clock change
        {
            std::lock_guard<std::mutex> lock(clockMutex);
            logical_clock++;
            cout << "Time:" << logical_clock << " sending update to " << clientid << endl;
        }
        string updateMessage = to_string(logical_clock) + " " + clientid + " " + to_string(expectedPosition.x);
        //string updateMessage = "Time:" + to_string(logical_clock) + " Client:" + clientid + " ExpectedX:" + to_string(expectedPosition.x);
        //cout << updateMessage << endl;
        int bytesSent = send(clientSocket, updateMessage.c_str(), updateMessage.length(), 0);

        if (bytesSent == SOCKET_ERROR && recoupleDuration != 15) {
            cerr << "Communication failure with follower: " << clientid << " attempting to reconnect" << endl;
            recoupleDuration = recoupleDuration + 1;
        } else {
            if (recoupleDuration == 15) {
                cerr << "Decoupling with follower: " << clientid << endl;
                std::lock_guard<std::mutex> leaderLock(leaderMutex);
                leader.removeFollower(clientid);
                cout << "Removed follower: " << clientid << endl;
                break;
            }
        }


        this_thread::sleep_for(chrono::seconds(1));
    }

    closesocket(clientSocket);
}

void incrementLogicalClockLeader() {
    while (true) {
        {
            std::lock_guard<std::mutex> leaderLock(leaderMutex);
            leader.runForSeconds(1);
        }

        cout << "INFO: " << leader.toString() << endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int LeaderTruckController() {

    leader.setID("LTRK012");
    leader.setPosition(Position{500, 1});
    leader.setSpeedCPP(60);

    std::thread clockThread(incrementLogicalClockLeader);
    clockThread.detach();

    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        cerr << "Failed to initialize Winsock." << endl;
        return EXIT_FAILURE;
    }

    // Create socket
    SOCKET serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == INVALID_SOCKET) {
        cerr << "Failed to create server socket." << endl;
        WSACleanup();
        return EXIT_FAILURE;
    }

    // Bind the socket
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(PORT);

    if (bind(serverSocket, (struct sockaddr *) &serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        cerr << "Bind failed." << endl;
        closesocket(serverSocket);
        WSACleanup();
        return EXIT_FAILURE;
    }

    // Listen for incoming connections
    if (listen(serverSocket, SOMAXCONN) == SOCKET_ERROR) {
        cerr << "Listen failed." << endl;
        closesocket(serverSocket);
        WSACleanup();
        return EXIT_FAILURE;
    }

    cout << "Server listening on port " << PORT << endl;

    vector<thread> clientThreads;

    while (true) {
        SOCKET clientSocket = accept(serverSocket, nullptr, nullptr);
        if (clientSocket == INVALID_SOCKET) {
            cerr << "Failed to create socket" << endl;
            closesocket(serverSocket);
            WSACleanup();
            return EXIT_FAILURE;
        }

        clientThreads.emplace_back(handleClient, clientSocket);
        clientThreads.back().detach();
    }

    // Cleanup
    closesocket(serverSocket);
    WSACleanup();

    return EXIT_SUCCESS;
}

int logical_follower_clock = 0;
mutex logicalMutex;


struct ParsedData {
    int time;
    std::string client;
    double expectedX;
};

ParsedData parseCharArray(const char *input) {
    ParsedData result;
    std::istringstream iss(input);

    // Parse Time, Client, and ExpectedX
    iss >> result.time >> result.client >> result.expectedX;

    return result;
}


SOCKET establishConnection() {
    // Create a new socket
    SOCKET newSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (newSocket == INVALID_SOCKET) {
        cerr << "Failed to recreate client socket." << endl;
        return INVALID_SOCKET;
    }

    // Server details
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(PORT);
    serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1");

    // Connect to the server
    if (connect(newSocket, (struct sockaddr *) &serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        cerr << "Failed to reconnect to the server." << endl;
        closesocket(newSocket);
        return INVALID_SOCKET;
    }

    char buffer[BUFFER_SIZE];
    int bytesRead = recv(newSocket, buffer, sizeof(buffer), 0);
    buffer[bytesRead] = '\0';
    string receivedData(buffer, bytesRead);
    string prefixToCheck = "Leader";
    if (receivedData.compare(0, prefixToCheck.length(), prefixToCheck) == 0) {
        string leaderId = receivedData.substr(prefixToCheck.length());
        cout << "Coupled with " << leaderId << endl;
        follower.setLeader(leaderId);
    }
    //resend join request
    if (follower.getLeader() != "") {
        //sending Join Request after coupling
        string joinMessage = "join:" + follower.getId() + ":[" + follower.getPositionString() + "]";
        send(newSocket, joinMessage.c_str(), joinMessage.length(), 0);
    }

    return newSocket;
}

void incrementLogicalClockFollower() {
    while (true) {

        {
            std::lock_guard<std::mutex> followerLock(followerMutex);
            follower.runForSeconds(1);

        }

        cout << "INFO: " << follower.toString() << endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int FollowerTruckController(string myid) {
    follower.setID(myid);
    follower.setPosition(Position{507.0, 0});

    std::thread followerClockThread(incrementLogicalClockFollower);
    followerClockThread.detach();

    // Initialize Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        cerr << "Failed to initialize Winsock." << endl;
        return EXIT_FAILURE;
    }

    // Create socket
    SOCKET clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == INVALID_SOCKET) {
        cerr << "Failed to create client socket." << endl;
        WSACleanup();
        return EXIT_FAILURE;
    }

    // Server details
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(PORT);

    // Convert IP address to binary form
    serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    if (serverAddr.sin_addr.s_addr == INADDR_NONE) {
        cerr << "Failed to convert IP address." << endl;
        closesocket(clientSocket);
        WSACleanup();
        return EXIT_FAILURE;
    }

    // Connect to the server
    if (connect(clientSocket, (struct sockaddr *) &serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        cerr << "Failed to find a leader to couple with" << endl;
        closesocket(clientSocket);
        WSACleanup();
        return EXIT_FAILURE;
    }


    char buffer[BUFFER_SIZE];
    int bytesRead = recv(clientSocket, buffer, sizeof(buffer), 0);
    if (bytesRead > 0) {
        buffer[bytesRead] = '\0';
        string receivedData(buffer, bytesRead);
        //cout << "Received from server: " << buffer << endl;
        string prefixToCheck = "Leader";
        if (receivedData.compare(0, prefixToCheck.length(), prefixToCheck) == 0) {
            //cout << "Received data starts with Leader." << endl;
            string leaderId = receivedData.substr(prefixToCheck.length());
            cout << "Coupled with leader" << leaderId << endl;
            follower.setLeader(leaderId);
        } else {
            cerr << "Coupling failed" << endl;
        }

        if (follower.getLeader() != "") {
            //sending Join Request after coupling
            string joinMessage = "join:" + follower.getId() + ":[" + follower.getPositionString() + "]";
            send(clientSocket, joinMessage.c_str(), joinMessage.length(), 0);
        }
        int reconnectAttempt = 0;
        while (true) {

            bytesRead = recv(clientSocket, buffer, sizeof(buffer), 0);

            if (bytesRead > 0) {
                buffer[bytesRead] = '\0';
                //cout << "From Leader: " << buffer << endl;
                ParsedData parsedData = parseCharArray(buffer);
                //cout << "ID:"<< parsedData.client <<" Speed:60 Position:x="<< to_string(parsedData.expectedX)<< ",y=1" << endl;
                //std::cout << "Time: " << parsedData.time << std::endl;
                //std::cout << "Client: " << parsedData.client << std::endl;
                //std::cout << "ExpectedX: " << parsedData.expectedX << std::endl;
                if (logical_follower_clock != parsedData.time) {
                    //synchronize time
                    cout << "Synchronizing time" << endl;
                    std::lock_guard<std::mutex> lockClock(logicalMutex);
                    logical_follower_clock = max(logical_follower_clock,parsedData.time);
                }
                {
                    std::lock_guard<std::mutex> lock(logicalMutex);
                    logical_follower_clock++;
                    cout << "Logical clock:"<<logical_follower_clock<<" "<<endl;
                }


                // PID implementation
                float pidError = static_cast<float>(parsedData.expectedX - follower.getPosition().x);
                // Call the PID function with errors, outputs, and constants
                float errors[1] = {pidError};
                float outputs[1] = {0.0f};  // Initialize with appropriate output values
                float constants[3] = {0.3f, 0.02f, 0.2f};  // Replace with your actual PID constants
                float pidResult = follower.cudaPidComputation(errors, outputs, constants, 1);
                //cout << "pid:" << pidResult << endl;
                int newSpeed = follower.getSpeed() + static_cast<int>(pidResult);
                int newSpeed;
                {
                //Use a lock_guard to protect access to shared data
                    std::lock_guard<std::mutex> followerLock(followerMutex);
                    int newSpeed;
                }
                {
                    std::lock_guard<std::mutex> lockSpeed(fSpdMutex);
                    newSpeed = follower_Speed;
                }

                float error;
                {
                    std::lock_guard<std::mutex> lockPos(fPosMutex);
                    error = parsedData.expectedX - follower_pos;
                }

                if (error > 0) {
                    newSpeed = newSpeed + 1;
                } else {
                    newSpeed = newSpeed - 1;
                }

                newSpeed = max(40, min(newSpeed, 80));

                //std::cout << "new speed" << newSpeed << std::endl;

                follower.setSpeed(80);
                follower.runForSeconds(1);
                }


                //std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            } else {
                closesocket(clientSocket);
                while (reconnectAttempt < 15) {
                    cerr << "Trying to reconnect, attempt " << reconnectAttempt + 1 << endl;
                    clientSocket = establishConnection();
                    if (clientSocket != INVALID_SOCKET) {
                        std::cout << "Recoupled to the server." << std::endl;
                        reconnectAttempt = 16;
                        break;
                    } else {
                        closesocket(clientSocket);
                        reconnectAttempt = reconnectAttempt + 1;
                        if (reconnectAttempt == 15) {
                            cerr << "Failed to recouple. Removing leader: " << follower.getLeader() << endl;
                            follower.setLeader("");
                            cerr << "Stopping truck..." << endl;
                            //add pid stop
                        }
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        continue;
                    }
                }
            }

        }
    }


    shutdown(clientSocket, SD_SEND);
    closesocket(clientSocket);

    WSACleanup();

    return EXIT_SUCCESS;
}

int main() {
    int choice;
    cout << "Select mode:\n";
    cout << "1. Leader\n";
    cout << "2. Follower\n";
    cout << "Enter choice: ";
    cin >> choice;

    if (choice == 1) {
        LeaderTruckController();
    } else if (choice == 2) {
        string id;
        cout << "Enter ID: ";
        cin >> id;
        FollowerTruckController(id);
    } else {
        cerr << "Invalid choice. Exiting." << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
