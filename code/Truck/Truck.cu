#include "Truck.cuh"
#include <string>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

Truck::Truck(string id) : id(id), state(TruckState::IDLE), speed(60), postion(), size(5), gap(2)
{
    postion.x = 0;
    postion.y = 0;
};

std::string Truck::getId()
{
    return id;
}

void Truck::setState(TruckState newState)
{
    state = newState;
}

TruckState Truck::getState()
{
    return state;
}

string Truck::getStateString()
{
    switch (state)
    {
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

__device__ int Truck::getSpeed()
{
    return speed;
}

int Truck::getSpeedCPP()
{
    return speed;
}

__device__ void Truck::setSpeed(int newSpeed)
{
    if (speed != 0 && newSpeed == 0)
    {
        state = TruckState::STOP;
    }
    else if (newSpeed > speed)
    {
        state = TruckState::ACCELERATE;
    }
    else if (newSpeed < speed)
    {
        state = TruckState::DECELERATE;
    }
    else if (newSpeed == speed && newSpeed != 0)
    {
        state = TruckState::CRUISE;
    }
    else
    {
        state = TruckState::IDLE;
    }

    speed = newSpeed;
}

void Truck::setSpeedCPP(int newSpeed)
{
    if (speed != 0 && newSpeed == 0)
    {
        state = TruckState::STOP;
    }
    else if (newSpeed > speed)
    {
        state = TruckState::ACCELERATE;
    }
    else if (newSpeed < speed)
    {
        state = TruckState::DECELERATE;
    }
    else if (newSpeed == speed && newSpeed != 0)
    {
        state = TruckState::CRUISE;
    }
    else
    {
        state = TruckState::IDLE;
    }

    speed = newSpeed;
}

Position Truck::getPosition()
{
    return postion;
}

string Truck::getPositionString()
{
    // format is "x:value, y:value"
    return "x:" + to_string(postion.x) + ",y:" + to_string(postion.y);
}
void Truck::runForSeconds(int second)
{
    Position temp = getPosition();
    for (int i = 0; i < second; i++)
    {
        temp.x = temp.x + (speed / 3.6);
    }
    setPosition(temp);
}

void Truck::setPosition(Position newPosition)
{
    postion.x = newPosition.x;
    postion.y = newPosition.y;
}

void Truck::setID(std::string newId)
{
    id = newId;
}

Position parsePosition(std::string positionString)
{
    Position result;
    std::istringstream iss(positionString);
    char discard;
    iss >> discard >> result.x >> discard >> discard >> result.y >> discard;

    return result;
}

void Truck::setGap(double newGap)
{
    gap = newGap;
}

void Truck::setSize(double newSize)
{
    size = newSize;
}

double Truck::getSize()
{
    return size;
}

double Truck::getGap()
{
    return gap;
}

const int numTrucks = 1;

// CUDA kernel for PID calculation
__global__ void pidKernel(Truck *trucks, float *errors, float *outputs, const float *constants, int numIterations)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < numIterations; ++i)
    {
        // Calculate PID components
        float proportional = constants[0] * errors[idx];
        float integral = constants[1] * errors[idx];
        float derivative = constants[2] * errors[idx];

        // Update output based on PID components
        outputs[idx] = proportional + integral + derivative;

        // Update truck state or speed based on the calculated output (this may vary based on your application)
        trucks[idx].setSpeed(trucks[idx].getSpeed() + static_cast<int>(outputs[idx]));
    }
}

float Truck::cudaPidComputation(float *errors, float *outputs, const float *constants, int numIterations)
{
    // Device variables
    float *d_errors;
    float *d_outputs;
    float *d_constants;
    Truck *d_trucks;

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_errors, numTrucks * sizeof(float));
    cudaMalloc((void **)&d_outputs, numTrucks * sizeof(float));
    cudaMalloc((void **)&d_constants, 3 * sizeof(float));
    cudaMalloc((void **)&d_trucks, numTrucks * sizeof(Truck));

    // Copy data from host to device
    cudaMemcpy(d_errors, errors, numTrucks * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_constants, constants, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trucks, this, numTrucks * sizeof(Truck), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    pidKernel<<<1, numTrucks>>>(d_trucks, d_errors, d_outputs, d_constants, numIterations);
    cudaDeviceSynchronize();

    // Copy results from device to host (if needed)
    cudaMemcpy(outputs, d_outputs, numTrucks * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy modified truck data back to host
    cudaMemcpy(this, d_trucks, numTrucks * sizeof(Truck), cudaMemcpyDeviceToHost);

    float result;
    for (int i = 0; i < numTrucks; ++i)
    {
        // std::cout << "Truck " << id << " PID Output: " << outputs[i] << std::endl;
        result = outputs[i];
    }

    // Cleanup
    cudaFree(d_errors);
    cudaFree(d_outputs);
    cudaFree(d_constants);
    cudaFree(d_trucks);

    return result;
}

string Truck::toString()
{
    return "ID:" + id + " Speed:" + to_string(speed) + " Pos:" + getPositionString();
}