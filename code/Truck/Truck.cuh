#ifndef TRUCK_PLATOONING_TRUCK_CUH
#define TRUCK_PLATOONING_TRUCK_CUH

#include <string>
#include <cuda_runtime.h>

enum class TruckState
{
    IDLE,
    CRUISE,
    ACCELERATE,
    DECELERATE,
    STOP
};

struct Position
{
    double x;
    int y;
};

class Truck
{
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
    __device__ int getSpeed();
    __device__ void setSpeed(int newSpeed);
    int getSpeedCPP();
    void setSpeedCPP(int newSpeed);
    Position getPosition();
    std::string getPositionString();
    void setPosition(Position newPosition);
    double getGap();
    void setGap(double newGap);
    double getSize();
    void setSize(double newSize);
    float cudaPidComputation(float *errors, float *outputs, const float *constants, int numIterations);
    std::string toString();
    void runForSeconds(int second);
};
Position parsePosition(std::string positionString);

#endif // TRUCK_PLATOONING_TRUCK_CUH