#define __SYCL_ENABLE_EXCEPTIONS
#define SYCL_

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

// #ifdef HIPSYCL
// #include <SYCL/sycl.hpp>
// #endif

// #ifdef DPCPP
// #include <CL/sycl.hpp>
// #endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util.hpp" // utility library
using namespace sycl;

#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5
#define BLOCK_SIZE_X_ID 6
#define BLOCK_SIZE_Y_ID 7

#define P_R 0.5
#define P_EPSILON 0.001
#define NUMBER_OF_OUTFLOWS 4
#define STRLEN 256

#define calSetMatrixElement(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define calGetMatrixElement(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define calSetBufferedMatrixElement(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define calGetBufferedMatrixElement(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

void readGISInfo(char* path, int &r, int &c, /*double &xllcorner, double &yllcorner, double &cellsize,*/ double &nodata)
{
  FILE* f;
  
  if ( (f = fopen(path,"r") ) == 0){
    printf("Configuration header file not found\n");
    exit(0);
  }

  char str[STRLEN];
  //int cont = -1;
  //fpos_t position;

  //Reading the header
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); c = atoi(str);         //ncols
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); r = atoi(str);         //nrows
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //xllcorner = atof(str); //xllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //yllcorner = atof(str); //yllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //cellsize = atof(str);  //cellsize
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nodata = atof(str);    //NODATA_value 

//  //Checks if actually there are ncols x nrows values into the file
//  fgetpos (f, &position);
//  while (!feof(f))
//  {
//    fscanf(f,"%s",&str);
//    cont++;
//  }
//  fsetpos (f, &position);
//  if (r * c != cont)
//  {
//    printf("File corrupted\n");
//    exit(0);
//  }
}

void calfLoadMatrix2Dr(double *M, int rows, int columns, FILE *f)
{
  char str[STRLEN];
  int i, j;

  for (i = 0; i < rows; i++)
    for (j = 0; j < columns; j++)
    {
      fscanf(f, "%s", str);
      calSetMatrixElement(M, columns, i, j, atof(str));
    }
}

bool calLoadMatrix2Dr(double *M, int rows, int columns, char *path)
{
  FILE *f = NULL;
  f = fopen(path, "r");

  if (!f)
    return false;

  calfLoadMatrix2Dr(M, rows, columns, f);

  fclose(f);

  return true;
}

void calfSaveMatrix2Dr(double *M, int rows, int columns, FILE *f)
{
  char str[STRLEN];
  int i, j;

  for (i = 0; i < rows; i++)
  {
    for (j = 0; j < columns; j++)
    {
      sprintf(str, "%f ", calGetMatrixElement(M, columns, i, j));
      fprintf(f, "%s ", str);
    }
    fprintf(f, "\n");
  }
}

bool calSaveMatrix2Dr(double *M, int rows, int columns, char *path)
{
  FILE *f;
  f = fopen(path, "w");

  if (!f)
    return false;

  calfSaveMatrix2Dr(M, rows, columns, f);

  fclose(f);

  return true;
}

double *calAddSingleLayerSubstate2Dr(int rows, int columns)
{

  double *tmp = (double *)malloc(sizeof(double) * rows * columns);
  if (!tmp)
    return NULL;
  return tmp;
}

void sciddicaTSimulationInit(int r, int c, double* Sz, double* Sh)
{
  double z, h;
  int i, j;

  for (i = 0; i < r; i++)
    for (j = 0; j < c; j++)
    {
      h = calGetMatrixElement(Sh, c, i, j);

      if (h > 0.0)
      {
        z = calGetMatrixElement(Sz, c, i, j);
        calSetMatrixElement(Sz, c, i, j, z - h);
      }
    }
}

int main(int argc, char **argv)
{
  int rows, cols;
  double  nodata;
  readGISInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

  int THREADS_PER_BLOCK_X = atoi(argv[BLOCK_SIZE_X_ID]);
  int THREADS_PER_BLOCK_Y = atoi(argv[BLOCK_SIZE_Y_ID]);

  double *Sz;
  double *Sh;
  //double *Sf;
  int r = rows;
  int c = cols;
  int i_start = 1, i_end = r-1;
  int j_start = 1, j_end = c-1;
  int Xi[] = {0, -1,  0,  0,  1};
  int Xj[] = {0,  0, -1,  1,  0};
  double p_r = P_R;
  double p_epsilon = P_EPSILON;
  int steps = atoi(argv[STEPS_ID]);

  Sz = calAddSingleLayerSubstate2Dr(r, c);
  Sh = calAddSingleLayerSubstate2Dr(r, c);
  
  calLoadMatrix2Dr(Sz, r, c, argv[DEM_PATH_ID]);
  calLoadMatrix2Dr(Sh, r, c, argv[SOURCE_PATH_ID]);
  sciddicaTSimulationInit(r, c, Sz, Sh);

  queue Q;
  double *d_Sz = malloc_device<double>(r*c, Q);
  double *d_Sh = malloc_device<double>(r*c, Q);
  double *d_Sf = malloc_device<double>(r*c*NUMBER_OF_OUTFLOWS, Q);

/*
  Q.submit([&](handler& h) {
      // copy hostArray to deviceArray
      h.memcpy(d_Sh, Sh, r*c * sizeof(double));
  });
  Q.wait(); // needed for now (we learn a better way later) 
  Q.submit([&](handler& h) {
      // copy hostArray to deviceArray
      h.memcpy(d_Sz, Sz, r*c * sizeof(double));
  });
  Q.wait(); // needed for now (we learn a better way later) 
*/

  Q.memcpy(d_Sh, Sh, r*c*sizeof(double)).wait();
  Q.memcpy(d_Sz, Sz, r*c*sizeof(double)).wait();

  // int NDRr = 512;
  // int NDRc = 640;
  int NDRr_L = THREADS_PER_BLOCK_X;
  int NDRc_L = THREADS_PER_BLOCK_Y;
  int NDRr = NDRr_L * ( ceil((float)r / NDRr_L ));
  int NDRc = NDRc_L * ( ceil((float)c / NDRc_L ));

  util::Timer cl_timer;
  for (int step = 0; step < steps; step++)
  {
      // reset kernel
      Q.submit([&](handler& h) {
          h.parallel_for<class Reset>(nd_range<2>(range<2>(NDRr,NDRc), range<2>(NDRr_L,NDRc_L)), [=](nd_item<2> idx) {

            // int i = idx[1];
            // int j = idx[0];
            int i = idx.get_global_id(0);
            int j = idx.get_global_id(1);

  	        if (i < 1 || i >= r-1 || j < 1 || j >= c-1)
	            return;

            calSetBufferedMatrixElement(d_Sf, r, c, 0, i, j, 0.0);
            calSetBufferedMatrixElement(d_Sf, r, c, 1, i, j, 0.0);
            calSetBufferedMatrixElement(d_Sf, r, c, 2, i, j, 0.0);
  	        calSetBufferedMatrixElement(d_Sf, r, c, 3, i, j, 0.0);
        });
      });

      // flow computation kernel
      Q.submit([&](handler& h) {
          h.parallel_for<class Compute>(nd_range<2>(range<2>(NDRr,NDRc), range<2>(NDRr_L,NDRc_L)), [=](nd_item<2> idx) {
            // int i = idx[1];
            // int j = idx[0];
            int i = idx.get_global_id(0);
            int j = idx.get_global_id(1);


            if (i < 1 || i >= r-1 || j < 1 || j >= c-1)
                return;

            //if (calGetMatrixElement(Sz, c, i, j) == nodata)
            //    return;

            int Xi[] = {0, -1,  0,  0,  1};
            int Xj[] = {0,  0, -1,  1,  0};

            bool eliminated_cells[5] = {false, false, false, false, false};
            bool again = false;
            int cells_count;
            double average = 0.0;
            double m;
            double u[5];
            int n;
            double z, h;

            m = calGetMatrixElement(d_Sh, c, i, j) - p_epsilon;
            u[0] = calGetMatrixElement(d_Sz, c, i, j) + p_epsilon;
            z = calGetMatrixElement(d_Sz, c, i + Xi[1], j + Xj[1]);
            h = calGetMatrixElement(d_Sh, c, i + Xi[1], j + Xj[1]);
            u[1] = z + h;                                         
            z = calGetMatrixElement(d_Sz, c, i + Xi[2], j + Xj[2]);
            h = calGetMatrixElement(d_Sh, c, i + Xi[2], j + Xj[2]);
            u[2] = z + h;                                         
            z = calGetMatrixElement(d_Sz, c, i + Xi[3], j + Xj[3]);
            h = calGetMatrixElement(d_Sh, c, i + Xi[3], j + Xj[3]);
            u[3] = z + h;                                         
            z = calGetMatrixElement(d_Sz, c, i + Xi[4], j + Xj[4]);
            h = calGetMatrixElement(d_Sh, c, i + Xi[4], j + Xj[4]);
            u[4] = z + h;

            do
            {
              again = false;
              average = m;
              cells_count = 0;

              for (n = 0; n < 5; n++)
                if (!eliminated_cells[n])
                {
                  average += u[n];
                  cells_count++;
                }

              if (cells_count != 0)
                average /= cells_count;

              for (n = 0; n < 5; n++)
                if ((average <= u[n]) && (!eliminated_cells[n]))
                {
                  eliminated_cells[n] = true;
                  again = true;
                }
            } while (again);

            if (!eliminated_cells[1]) calSetBufferedMatrixElement(d_Sf, r, c, 0, i, j, (average - u[1]) * p_r);
            if (!eliminated_cells[2]) calSetBufferedMatrixElement(d_Sf, r, c, 1, i, j, (average - u[2]) * p_r);
            if (!eliminated_cells[3]) calSetBufferedMatrixElement(d_Sf, r, c, 2, i, j, (average - u[3]) * p_r);
            if (!eliminated_cells[4]) calSetBufferedMatrixElement(d_Sf, r, c, 3, i, j, (average - u[4]) * p_r);	
          });
      });
  
      // balance kernel
      Q.submit([&](handler& h) {
          h.parallel_for<class Update>(nd_range<2>(range<2>(NDRr,NDRc), range<2>(NDRr_L,NDRc_L)), [=](nd_item<2> idx) {
            // int i = idx[1];
            // int j = idx[0];
            int i = idx.get_global_id(0);
            int j = idx.get_global_id(1);

            if (i < 1 || i >= r-1 || j < 1 || j >= c-1)
                return;

            //if (calGetMatrixElement(Sz, c, i, j) == nodata)
            //    return;

            int Xi[] = {0, -1,  0,  0,  1};
            int Xj[] = {0,  0, -1,  1,  0};

            double h_next;
            h_next = calGetMatrixElement(d_Sh, c, i, j);
            h_next += calGetBufferedMatrixElement(d_Sf, r, c, 3, i+Xi[1], j+Xj[1]) - calGetBufferedMatrixElement(d_Sf, r, c, 0, i, j);
            h_next += calGetBufferedMatrixElement(d_Sf, r, c, 2, i+Xi[2], j+Xj[2]) - calGetBufferedMatrixElement(d_Sf, r, c, 1, i, j);
            h_next += calGetBufferedMatrixElement(d_Sf, r, c, 1, i+Xi[3], j+Xj[3]) - calGetBufferedMatrixElement(d_Sf, r, c, 2, i, j);
            h_next += calGetBufferedMatrixElement(d_Sf, r, c, 0, i+Xi[4], j+Xj[4]) - calGetBufferedMatrixElement(d_Sf, r, c, 3, i, j);

            calSetMatrixElement(d_Sh, c, i, j, h_next);          	
                    });
      });
   }
   Q.wait();
   double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
   printf("Elapsed time: %lf [s]\n", cl_time);

   /*Q.submit([&](handler& h) {
      // copy hostarray to devicearray
      h.memcpy(Sh, d_Sh, r*c * sizeof(double));
   });
*/
   Q.memcpy(Sh, d_Sh, r*c*sizeof(double)).wait();

   calSaveMatrix2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]);
  
   /*** Releasing memory ***/
   printf("Releasing memory...\n");
   free(d_Sh, Q);
   free(d_Sz, Q);
   free(d_Sf, Q);
   delete[] Sz;
   delete[] Sh;

  return 0;
}
