#ifndef TRUCK_PLATOONING_LEADERTRUCK_H
#define TRUCK_PLATOONING_LEADERTRUCK_H

#include "../Truck.h"
#include <string>
#include <vector>

class LeaderTruck: public Truck{
private:
    std::string port;
    std::vector<std::string> followers;
public:
    LeaderTruck(std::string id);
    std::string  getPort();
    void setPort(std::string myport);
    int checkIfFollower(std::string followerId);
    void addFollower(std::string newFollower);
    void removeFollower(std::string followerId);

};

#endif //TRUCK_PLATOONING_LEADERTRUCK_H
