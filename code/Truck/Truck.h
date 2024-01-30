#ifndef TRUCK_PLATOONING_TRUCK_H
#define TRUCK_PLATOONING_TRUCK_H

#include <string>

enum class TruckState {
    IDLE,
    CRUISE,
    ACCELERATE,
    DECELERATE,
    STOP
};

struct Position {
    double x;
    int y;
};


class Truck{
private:
    std::string id;
    TruckState state;
    int speed;
    Position postion;
    double size;
    double gap;
public:
    Truck(std::string id);
    std::string getId();
    void setID(std::string newId);
    void setState(TruckState newState);
    TruckState getState();
    std::string getStateString();
    int getSpeed();
    void setSpeed(int newSpeed);
    Position getPosition();
    std::string getPositionString();
    void setPosition(Position newPosition);
    double getGap();
    void setGap(double newGap);
    double getSize();
    void setSize(double newSize);

};
Position parsePosition(std::string positionString);

#endif //TRUCK_PLATOONING_TRUCK_H
