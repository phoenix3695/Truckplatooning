
#ifndef TRUCK_PLATOONING_FOLLOWERTRUCK_H
#define TRUCK_PLATOONING_FOLLOWERTRUCK_H

#include <string>
#include "../Truck.h"

class FollowerTruck: public Truck {
private:
    std::string port;
    std::string leader;


public:
    FollowerTruck(std::string id);
    std::string  getPort();
    void setPort(std::string myport);
    void setLeader(std::string newLeader);
    std::string getLeader();
};



#endif //TRUCK_PLATOONING_FOLLOWERTRUCK_H
