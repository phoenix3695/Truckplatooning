#include "LeaderTruck.h"
#include <string>
#include <algorithm>

using namespace std;

LeaderTruck::LeaderTruck(string id):Truck(id){};

void LeaderTruck::setPort(std::string myport) {
    port = myport;
};

string LeaderTruck::getPort() {
  return port;
};

int LeaderTruck::checkIfFollower(std::string followerId) {
    auto it = std::find(followers.begin(), followers.end(), followerId);
    if (it != followers.end()) {
        return std::distance(followers.begin(), it);
    } else {
        return -1;
    }
}

void LeaderTruck::addFollower(std::string newFollower) {
    followers.push_back(newFollower);
}

void LeaderTruck::removeFollower(std::string followerId) {
    auto it = std::find(followers.begin(), followers.end(), followerId);
    if (it != followers.end()) {
        followers.erase(it);
    }
}