/*********************************************************************************************************************
* file_handling
* -------------------
* This library continains functions to perform file input/output operations
*
* Author: Thomas Fairbank <tfairabn@myune.edu.au>
* Student ID: 205167657
*
*********************************************************************************************************************/
#include "imu.h"

/*********************************************************************************************************************
* parseCSVmeta
* -------------------
* Parses a CSV of specified number of columns and returns an integer value for the number of rows in the file or error
*
* Inputs:
* file - path to CSV to open
* column - integer value for the expected number of columns
*
*********************************************************************************************************************/
int parseCSVmeta (const char *file, int column){
    int lines = 0; //line counter
    int colcount = 0; //column counter
    char c;
    FILE *fp;
    fp = fopen(file, "r");
    if (!fp) {
      fprintf(stderr, FOPEN_ERROR, file);
      exit(EXIT_FAILURE);
    }
    for (c = getc(fp); c != EOF; c = getc(fp)){
        if (c == ',') {
            colcount++;
        }
        if (c == '\n') {
            lines++;
            if (colcount + 1 != column) {
              fprintf(stderr, "Unexpected number of columns on line: %d\nFound: %d\n"
              "Expected: %d\nAborting.\n", lines, colcount, column);
              exit(EXIT_FAILURE);
            }
            colcount = 0; //reset the column count to zero for next iteration
        }
    }
        // Close the file
        fclose(fp);
        return lines;
}

/*********************************************************************************************************************
* parseCSV
* -------------------
* Parses a CSV of 3 columns into specified (float) arrays
*
* Inputs:
* file - path to CSV to open
*
* Outputs:
* ax - float array output for first value
* ay - float array output for second value
* az - float array output for Third value
*
*********************************************************************************************************************/
void parseCSV (const char *file, float *ax, float *ay, float *az){
    char buf[1024];
    int i = 0; /* count for data read*/
    FILE *fp;

    fp = fopen(file, "r");

    if (!fp) {
      fprintf(stderr, FOPEN_ERROR, file);
      exit(EXIT_FAILURE);
    }

    while (fgets(buf, sizeof(buf), fp) != NULL) {

    // read values into each array
    if (sscanf(buf, "%f,%f,%f", &ax[i], &ay[i], &az[i]) != 3) {
      fprintf(stderr, "sscanf failed.\n");
      exit(EXIT_FAILURE);
    }
    i++;
  }
  fclose(fp);
}

/*********************************************************************************************************************
* writeCSV
* -------------------
* Overwrite existing or creates a new CSV of 14 columns and specified number of rows using data from specified arrays
*
* Inputs:
* file - path to CSV to open
* rows - integer value for the number of rows to output
* sma - array of floats to be written as column 1
* mia - array of floats to be written as column 2
* sd_ax - array of floats to be written as column 3
* sd_ay - array of floats to be written as column 4
* sd_az - array of floats to be written as column 5
* mean_ax - array of floats to be written as column 6
* mean_ay - array of floats to be written as column 7
* mean_az - array of floats to be written as column 8
* max_ax - array of floats to be written as column 9
* max_ay - array of floats to be written as column 10
* max_az - array of floats to be written as column 11
* min_ax - array of floats to be written as column 12
* min_ay - array of floats to be written as column 13
* min_az - array of floats to be written as column 14
*
*********************************************************************************************************************/
void writeCSV (const char *file, int rows, float *sma, float *mia, float *sd_ax, float *sd_ay, float *sd_az,
                float *mean_ax, float *mean_ay, float *mean_az, float *max_ax, float *max_ay, float *max_az,
                float *min_ax,float *min_ay, float *min_az){

    FILE *fp = fopen(file, "w+");
    if (!fp) {
      fprintf(stderr, FOPEN_ERROR, file);
      exit(EXIT_FAILURE);
    }
    for (int i =0; i< rows; i++){
        fprintf(fp,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", sma[i], mia[i], sd_ax[i], sd_ay[i], sd_ax[i],
            mean_ax[i], mean_ay[i], mean_az[i], max_ax[i], max_ay[i], max_az[i], min_ax[i], min_ay[i], min_az[i]);
    }
    fclose(fp);
}
