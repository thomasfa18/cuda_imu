/*********************************************************************************************************************
* math_functions
* -------------------
* This library continains a CUDA kernel for calculating math fuctions across input vectors
*
* Author: Thomas Fairbank <tfairabn@myune.edu.au>
* Student ID: 205167657
*
*********************************************************************************************************************/
#include "imu.h"
/*********************************************************************************************************************
* CUDA Kernel: vectorEverything
* ----------------
*
* Implements vector simple arithmetic functions over a sequential "window" of data and stores results
* into appropriate output vector for each fucntion. All vecotrs are same length (vlength)
*
* Inputs:
* vlength - integer value of vector length
* window - integer value for the sequential window size
* X - Input vector X
* Y - Input vector Y
* Z - Input vector Z
* SMA - Output vector Signal Magnitude Area [(1/window)*(summations of vector absolute values of window)]
* MIA - Output vector Average Movement Intensity [(1/window)*(summations of addition of the vector squares of window)]
* SD_AX - Output vector Standard Deviation X [square root of the summation of (vector - window mean) squared of window]
* SD_AY - Output vector Standard Deviation Y [square root of the summation of (vector - window mean) squared of window]
* SD_AZ - Output vector Standard Deviation Z [square root of the summation of (vector - window mean) squared of window]
* MEAN_AX - Output vector Mean X [(1/window)*(summation of vector of window)]
* MEAN_AY - Output vector Mean Y [(1/window)*(summation of vector of window)]
* MEAN_AZ - Output vector Mean Z [(1/window)*(summation of vector of window)]
* MAX_AX - Output vector Max X [higest numeric value vector of the window]
* MAX_AY - Output vector Max Y [higest numeric value vector of the window]
* MAX_AZ - Output vector Max Z [higest numeric value vector of the window]
* MIN_AX - Output vector Min X [lowest numeric value vector of the window]
* MIN_AY - Output vector Min Y [lowest numeric value vector of the window]
* MIN_AZ - Output vector Min Z [lowest numeric value vector of the window]
*
*********************************************************************************************************************/

__global__ void vectorEverything(int vlength, int window, const float *X, const float *Y, const float *Z, float *SMA,
                                float *MIA, float *SD_AX, float *SD_AY, float *SD_AZ, float *MEAN_AX, float *MEAN_AY,
                                float *MEAN_AZ, float *MAX_AX, float *MAX_AY, float *MAX_AZ, float *MIN_AX,
                                float *MIN_AY, float *MIN_AZ)
{
    float reciprocal = 1/(float)window;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float sm = 0, mi = 0, sdx = 0, sdy = 0, sdz = 0, avx = 0, avy = 0, avz = 0, maxx = 0, maxy = 0, maxz = 0,
            minx = 0, miny=0, minz=0;
    if (i < vlength)
    {
        /* could add a unroll here but the compiler optimizes this pretty well aready, only 4us difference */
        /* #pragma unroll */

        for(int j=0; j < window; j++)
        {
            /* temp Signal Magnitude Area */
            sm += abs(X[i+j]) + abs(Y[i+j]) + abs(Z[i+j]);

            /* temp Average Movement Intensity */
            mi += (X[i+j]*X[i+j]) + (Y[i+j]*Y[i+j]) + (Z[i+j]*Z[i+j]);

            /* Max and Min */
            /* logic: the first value is always a conditional match */
            if (j == 0){
                maxx = X[i+j];
                maxy = Y[i+j];
                maxz = Z[i+j];
                minx = X[i+j];
                miny = Y[i+j];
                minz = Z[i+j];
            }

            /* replace the current value only if higher for Max and lower for Min */
            if (X[i+j] >= maxx){
                maxx = X[i+j];
            }
            if (Y[i+j] >= maxy){
                maxy = Y[i+j];
            }
            if (Z[i+j] >= maxz){
                maxz = Z[i+j];
            }
            if (X[i+j] <= minx){
                minx = X[i+j];
            }
            if (Y[i+j] <= miny){
                miny = Y[i+j];
            }
            if (Z[i+j] <= minz){
                minz = Z[i+j];
            }

            /* temp Mean */
            avx += X[i+j];
            avy += Y[i+j];
            avz += Z[i+j];
        }
        /* Final Mean */
        avx = reciprocal*avx;
        avy = reciprocal*avy;
        avz = reciprocal*avz;

        /* now that mean is known temp Standard Deviations can be calculated*/
        for(int j=0; j < window; j++)
        {
            sdx += (X[i+j]-avx)*(X[i+j]-avx);
            sdy += (Y[i+j]-avy)*(Y[i+j]-avy);
            sdz += (Z[i+j]-avz)*(Z[i+j]-avz);
        }

        /* Finalize temp calculations and store values in appropriate output vector */
        SMA[i] = reciprocal*sm;
        MIA[i] = reciprocal*mi;
        SD_AX[i] = sqrtf(sdx);
        SD_AY[i] = sqrtf(sdy);
        SD_AZ[i] = sqrtf(sdz);
        MEAN_AX[i] = avx;
        MEAN_AY[i] = avy;
        MEAN_AZ[i] = avz;
        MAX_AX[i] = maxx;
        MAX_AY[i] = maxy;
        MAX_AZ[i] = maxz;
        MIN_AX[i] = minx;
        MIN_AY[i] = miny;
        MIN_AZ[i] = minz;
    }
}
