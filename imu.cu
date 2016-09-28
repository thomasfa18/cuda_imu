/*********************************************************************************************************************
* imu
* -------------------
* This program implements GPU parallel programming via a CUDA kernel to process a csv containing x, y, and z
* acceleration data values recored by an Inertial Measurement Unit and calculate Signal Magnitude Area,
* Average Movement Intensity, Standard Deviation of each axis, Mean of each axis, min of each axis, and
* max of each axis along a 5 second sliding window.
*
* Relevant definitions imu.h header:
* WINDOW_LENGTH - integer (number of elements for 5 seconds)
* INFILE - path to the CSV to be read
* OUTFILE - path tot he CSV to be written
*
* Author: Thomas Fairbank <tfairabn@myune.edu.au>
* Student ID: 205167657
*
*********************************************************************************************************************/
#include "imu.h"

int main(void)
{
    /* initialize clock for single process run time (mem allocations and file I/O) */
    clock_t begin = clock ();

    /* initialize a variable for the Length in rows of complete data set*/
    int dataLength = 0;

    /* initialize variables for host arrays */
    float *ax_arr, *ay_arr, *az_arr, *sma_arr, *mia_arr, *sd_ax_arr,*sd_ay_arr, *sd_az_arr, *mean_ax_arr, *mean_ay_arr,
        *mean_az_arr, *max_ax_arr, *max_ay_arr, *max_az_arr, *min_ax_arr, *min_ay_arr, *min_az_arr;

    /* initialize timing variables and events for cuda processing operation */
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* determine the number of rows int he csv and set the dataLength (number of array elements) to this */
    dataLength = parseCSVmeta(INFILE,CSV_ROW_WIDTH);
    /* calculate the memory size required to store an array of this many elements */
    size_t size = dataLength * sizeof(float);

    /* allocate memory for host arrays */
    ax_arr  = (float *)malloc(size);
    ay_arr  = (float *)malloc(size);
    az_arr  = (float *)malloc(size);
    sma_arr = (float *)malloc(size);
    mia_arr = (float *)malloc(size);
    sd_ax_arr = (float *)malloc(size);
    sd_ay_arr = (float *)malloc(size);
    sd_az_arr = (float *)malloc(size);
    mean_ax_arr = (float *)malloc(size);
    mean_ay_arr = (float *)malloc(size);
    mean_az_arr = (float *)malloc(size);
    max_ax_arr = (float *)malloc(size);
    max_ay_arr = (float *)malloc(size);
    max_az_arr = (float *)malloc(size);
    min_ax_arr = (float *)malloc(size);
    min_ay_arr = (float *)malloc(size);
    min_az_arr = (float *)malloc(size);

    /*check that host array memory allocation was successful */
    if (ax_arr == NULL || ay_arr == NULL || az_arr == NULL || sma_arr == NULL || mia_arr == NULL || sd_ax_arr == NULL ||
        sd_ay_arr == NULL || sd_az_arr == NULL || mean_ax_arr == NULL || mean_ay_arr == NULL || mean_az_arr == NULL ||
        max_ax_arr == NULL || max_ay_arr == NULL || max_az_arr == NULL || min_ax_arr == NULL || min_ay_arr == NULL ||
        min_az_arr == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    /* read the file contents and split columns into the 3 arrays for x,y,z data */
    parseCSV (INFILE, ax_arr, ay_arr, az_arr);

    /* Allocate the device input vector X, Y, Z */
    float *d_X = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_X, size));

    float *d_Y = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Y, size));

    float *d_Z = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Z, size));

    /* Allocate the device output vectors O */
    float *d_Osma = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Osma, size));

    float *d_Omia = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Omia, size));

    float *d_Osdx = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Osdx, size));

    float *d_Osdy = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Osdy, size));

    float *d_Osdz = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Osdz, size));

    float *d_Oavx = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Oavx, size));

    float *d_Oavy = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Oavy, size));

    float *d_Oavz = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Oavz, size));

    float *d_Omaxx = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Omaxx, size));

    float *d_Omaxy = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Omaxy, size));

    float *d_Omaxz = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Omaxz, size));

    float *d_Ominx = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Ominx, size));

    float *d_Ominy = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Ominy, size));

    float *d_Ominz = NULL;
    errCheckCUDA(cudaMalloc((void **)&d_Ominz, size));

    /* calculate block size base on threads and dataLength */
    int threadsPerBlock = 256;
    int blocksPerGrid =(dataLength + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    /* Copy the host input vectors for x, y, z in host memory to the device input vectors in device memory */
    printf("Copy X acceleration data from the host memory to the CUDA device\n");
    errCheckCUDA(cudaMemcpy(d_X, ax_arr, size, cudaMemcpyHostToDevice));

    printf("Copy X acceleration data from the host memory to the CUDA device\n");
    errCheckCUDA(cudaMemcpy(d_Y, ay_arr, size, cudaMemcpyHostToDevice));

    printf("Copy Z acceleration data from the host memory to the CUDA device\n");
    errCheckCUDA(cudaMemcpy(d_Z, az_arr, size, cudaMemcpyHostToDevice));

    /* start cuda processing timer */
    cudaEventRecord(start,0);

    /* Launch the vectorEverything cuda kernel */
    vectorEverything<<<blocksPerGrid, threadsPerBlock>>>(dataLength, WINDOW_LENGTH, d_X, d_Y, d_Z, d_Osma, d_Omia,
            d_Osdx, d_Osdy, d_Osdz, d_Oavx, d_Oavy, d_Oavz, d_Omaxx, d_Omaxy, d_Omaxz, d_Ominx, d_Ominy, d_Ominz);

    /* stop cuda processing timer */
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    /* Calcualte cuda processing time */
    cudaEventElapsedTime(&elapsedTime,start,stop);

    printf("Copy sma data from the CUDA device to the host memory\n");
    errCheckCUDA(cudaMemcpy(sma_arr, d_Osma, size, cudaMemcpyDeviceToHost));

    printf("Copy mia data from the CUDA device to the host memory\n");
    errCheckCUDA(cudaMemcpy(mia_arr, d_Omia, size, cudaMemcpyDeviceToHost));

    printf("Copy sdx data from the CUDA device to the host memory\n");
    errCheckCUDA(cudaMemcpy(sd_ax_arr, d_Osdx, size, cudaMemcpyDeviceToHost));

    printf("Copy sdy data from the CUDA device to the host memory\n");
    errCheckCUDA(cudaMemcpy(sd_ay_arr, d_Osdy, size, cudaMemcpyDeviceToHost));

    printf("Copy sdz data from the CUDA device to the host memory\n");
    errCheckCUDA(cudaMemcpy(sd_az_arr, d_Osdz, size, cudaMemcpyDeviceToHost));

    printf("Copy avx data from the CUDA device to the host memory\n");
    errCheckCUDA(cudaMemcpy(mean_ax_arr, d_Oavx, size, cudaMemcpyDeviceToHost));

    printf("Copy avy data from the CUDA device to the host memory\n");
    errCheckCUDA(cudaMemcpy(mean_ay_arr, d_Oavy, size, cudaMemcpyDeviceToHost));

    printf("Copy avz data from the CUDA device to the host memory\n");
    errCheckCUDA(cudaMemcpy(mean_az_arr, d_Oavz, size, cudaMemcpyDeviceToHost));

    printf("Copy maxx data from the CUDA device to the host memory\n");
    errCheckCUDA(cudaMemcpy(max_ax_arr, d_Omaxx, size, cudaMemcpyDeviceToHost));

    printf("Copy maxy data from the CUDA device to the host memory\n");
    errCheckCUDA(cudaMemcpy(max_ay_arr, d_Omaxy, size, cudaMemcpyDeviceToHost));

    printf("Copy maxz data from the CUDA device to the host memory\n");
    errCheckCUDA(cudaMemcpy(max_az_arr, d_Omaxz, size, cudaMemcpyDeviceToHost));

    printf("Copy minx data from the CUDA device to the host memory\n");
    errCheckCUDA(cudaMemcpy(min_ax_arr, d_Ominx, size, cudaMemcpyDeviceToHost));

    printf("Copy miny data from the CUDA device to the host memory\n");
    errCheckCUDA(cudaMemcpy(min_ay_arr, d_Ominy, size, cudaMemcpyDeviceToHost));

    printf("Copy minz data from the CUDA device to the host memory\n");
    errCheckCUDA(cudaMemcpy(min_az_arr, d_Ominz, size, cudaMemcpyDeviceToHost));

    /* write output vectors to file */
    writeCSV (OUTFILE, dataLength - WINDOW_LENGTH, sma_arr, mia_arr, sd_ax_arr, sd_ay_arr, sd_az_arr, mean_ax_arr,
            mean_ay_arr, mean_az_arr, max_ax_arr, max_ay_arr, max_az_arr, min_ax_arr, min_ay_arr, min_az_arr);


    /* Free device global memory */
    errCheckCUDA(cudaFree(d_X));
    errCheckCUDA(cudaFree(d_Y));
    errCheckCUDA(cudaFree(d_Z));
    errCheckCUDA(cudaFree(d_Osma));
    errCheckCUDA(cudaFree(d_Omia));
    errCheckCUDA(cudaFree(d_Osdx));
    errCheckCUDA(cudaFree(d_Osdy));
    errCheckCUDA(cudaFree(d_Osdz));
    errCheckCUDA(cudaFree(d_Oavx));
    errCheckCUDA(cudaFree(d_Oavy));
    errCheckCUDA(cudaFree(d_Oavz));
    errCheckCUDA(cudaFree(d_Omaxx));
    errCheckCUDA(cudaFree(d_Omaxy));
    errCheckCUDA(cudaFree(d_Omaxz));
    errCheckCUDA(cudaFree(d_Ominx));
    errCheckCUDA(cudaFree(d_Ominy));
    errCheckCUDA(cudaFree(d_Ominz));

    /* Remember to play nice and Reset CUDA device */
    errCheckCUDA(cudaDeviceReset());

    /* free host memory */
    free(ax_arr);
    free(ay_arr);
    free(az_arr);
    free(sma_arr);
    free(mia_arr);
    free(sd_ax_arr);
    free(sd_ay_arr);
    free(sd_az_arr);
    free(mean_ax_arr);
    free(mean_ay_arr);
    free(mean_az_arr);
    free(max_ax_arr);
    free(max_ay_arr);
    free(max_az_arr);
    free(min_ax_arr);
    free(min_ay_arr);
    free(min_az_arr);

    /* output some debug information */
    fprintf(stderr,"Processing time breakdown:\n");
    /* print the cuda kernel processing time */
    fprintf(stderr,"CUDA kernel (vector functions) = %fms\n", elapsedTime);

    /* print the process run time */
    clock_t end = clock ();
    double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
    printf ("Non-CUDA (file and memory I/O) = %fs\n", time_spent);

    exit(EXIT_SUCCESS);
}
