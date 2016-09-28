#ifndef IMU_H
#define IMU_H

#include <stdio.h>  /* For I/O */
#include <stdlib.h>  /* For malloc operations */
#include <time.h>  /*For clock operations */
#include <cuda_runtime.h> /* For CUDA operations */
#include <math.h> /* For sqrtf */

/* file_handling macro definition */
#define INFILE "./data/sheep_imu_data.csv"
#define OUTFILE "./data/output.csv"
#define FOPEN_ERROR "Could not open file: %s\n"

#define CSV_ROW_WIDTH 3  /* Expected number of columns in CSV */
#define WINDOW_LENGTH 25 /* size of sliding window of data */

/* macro definition for CUDA error chcking */
#define errCheckCUDA(call) {                                                            \
    cudaError_t err = call;                                                             \
    if (err != cudaSuccess)                                                             \
    {                                                                                   \
        fprintf(stderr, "CUDA call failed\nError code: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                                             \
    }                                                                                   \
}                                                                                       \

/* math_functions function declarations*/
__global__ void vectorEverything(int vlength, int window, const float *X, const float *Y, const float *Z, float *SMA,
                                float *MIA, float *SD_AX, float *SD_AY, float *SD_AZ, float *MEAN_AX, float *MEAN_AY,
                                float *MEAN_AZ, float *MAX_AX, float *MAX_AY, float *MAX_AZ, float *MIN_AX,
                                float *MIN_AY, float *MIN_AZ);

/* file_handling function declarations */
int parseCSVmeta (const char *file, int column);
void parseCSV (const char *file, float *ax, float *ay, float *az);
void writeCSV (const char *file, int rows, float *sma, float *mia, float *sd_ax, float *sd_ay, float *sd_az, float *mean_ax,
                float *mean_ay, float *mean_az, float *max_ax, float *max_ay, float *max_az, float *min_ax,
                float *min_ay, float *min_az);

#endif
