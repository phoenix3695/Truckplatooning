#include "FollowerTruck.h"
#include <string>

using namespace std;

FollowerTruck::FollowerTruck(string id) : Truck(id), leader("") {};

void FollowerTruck::setPort(std::string myport) {
    port = myport;
};

string FollowerTruck::getPort() {
    return port;
};

string FollowerTruck::getLeader() {
    return leader;
}

void FollowerTruck::setLeader(std::string newLeader) {
    leader = newLeader;
}