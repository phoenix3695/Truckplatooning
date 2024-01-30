#include "Truck.h"
#include <string>
#include <sstream>

using namespace std;

Truck::Truck(string id) : id(id), state(TruckState::IDLE), speed(0), postion(), size(5), gap(2) {
    postion.x = 0;
    postion.y = 0;
};

std::string Truck::getId() {
    return id;
}

void Truck::setState(TruckState newState) {
    state = newState;
}

TruckState Truck::getState() {
    return state;
}

string Truck::getStateString() {
    switch (state) {
        case TruckState::IDLE:
            return "IDLE";
        case TruckState::CRUISE:
            return "CRUISE";
        case TruckState::ACCELERATE:
            return "ACCELERATE";
        case TruckState::DECELERATE:
            return "DECELERATE";
        case TruckState::STOP:
            return "STOP";
        default:
            return "";
    };
}

int Truck::getSpeed() {
    return speed;
}

void Truck::setSpeed(int newSpeed) {
    if (speed != 0 && newSpeed == 0) {
        state = TruckState::STOP;
    } else if (newSpeed > speed) {
        state = TruckState::ACCELERATE;
    } else if (newSpeed < speed) {
        state = TruckState::DECELERATE;
    } else if (newSpeed == speed && newSpeed != 0) {
        state = TruckState::CRUISE;
    } else {
        state = TruckState::IDLE;
    }

    speed = newSpeed;
}

Position Truck::getPosition() {
    return postion;
}

string Truck::getPositionString() {
    // format is "x:value, y:value"
    return "x:" + to_string(postion.x) + ",y:" + to_string(postion.y);
}

void Truck::setPosition(Position newPosition) {
    postion.x = newPosition.x;
    postion.y = newPosition.y;
}

void Truck::setID(std::string newId) {
    id = newId;
}

Position parsePosition(std::string positionString) {
    Position result;
    std::istringstream iss(positionString);
    char discard;
    iss >> discard >> result.x >> discard >> discard >> result.y >> discard;

    return result;
}

void Truck::setGap(double newGap) {
    gap = newGap;
}

void Truck::setSize(double newSize) {
    size = newSize;
}

double Truck::getSize() {
    return size;
}

double Truck::getGap() {
    return gap;
}