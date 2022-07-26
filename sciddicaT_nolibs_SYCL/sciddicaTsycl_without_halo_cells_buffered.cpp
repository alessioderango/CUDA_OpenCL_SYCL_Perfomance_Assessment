#define __SYCL_ENABLE_EXCEPTIONS
#define SYCL_

#ifdef HIPSYCL
#include <SYCL/sycl.hpp>
#endif

#ifdef DPCPP
#include <CL/sycl.hpp>
#endif

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
#define TILE_SIZE_ID 6

#define P_R 0.5
#define P_EPSILON 0.001
#define NUMBER_OF_OUTFLOWS 4
#define STRLEN 256

#define calSetMatrixElement(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define calGetMatrixElement(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define calSetBufferedMatrixElement(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define calGetBufferedMatrixElement(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

#define TILE_WIDTH_TILING_WIDTH 16
#define OFFSET 1
#define GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET (i+OFFSET)
#define GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET (j+OFFSET)
#define ADJACENT_CELLS 4

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
  
  int TILE_SIZE = atoi(argv[TILE_SIZE_ID]);
  
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
  auto d_Sz = new buffer<double>(Sz, r*c);
  auto d_Sh = new buffer<double>(Sh, r*c);
  auto d_Sf = new buffer<double>(r*c*NUMBER_OF_OUTFLOWS);

  const int size = r*c;
  util::Timer cl_timer;
  // int NDRr = 512;
  // int NDRc = 640;
  int NDRr_L = TILE_SIZE;
  int NDRc_L = TILE_SIZE;
  int NDRr = NDRr_L * ( ceil((float)r / NDRr_L ));
  int NDRc = NDRc_L * ( ceil((float)c / NDRc_L ));

  
  for (int step = 0; step < steps; step++)
  {
      // reset kernel
      Q.submit([&](handler& h) {
          auto a_d_Sf = d_Sf->template get_access<access::mode::write>(h);

          h.parallel_for<class Reset>(nd_range<2>(range<2>(NDRr,NDRc), range<2>(NDRr_L,NDRc_L)), [=](nd_item<2> idx) {
            // int i = idx[1];
            // int j = idx[0];
            int i = idx.get_global_id(0);
            int j = idx.get_global_id(1);

  	        if (i < 1 || i >= r-1 || j < 1 || j >= c-1)
	            return;

            calSetBufferedMatrixElement(a_d_Sf, r, c, 0, i, j, 0.0);
            calSetBufferedMatrixElement(a_d_Sf, r, c, 1, i, j, 0.0);
            calSetBufferedMatrixElement(a_d_Sf, r, c, 2, i, j, 0.0);
  	        calSetBufferedMatrixElement(a_d_Sf, r, c, 3, i, j, 0.0);
        });
      });

      // flow computation kernel
      Q.submit([&](handler& h) {
          auto a_d_Sf = d_Sf->template get_access<access::mode::write>(h);
          auto a_d_Sz = d_Sz->template get_access<access::mode::read>(h);
          auto a_d_Sh = d_Sh->template get_access<access::mode::read>(h);

          auto u_neighbour_ds = accessor<double, 2, access::mode::read_write, access::target::local>(sycl::range<2>(NDRr_L,NDRc_L), h);

          // accessor<double, 2, access::mode::read_write, access::target::local> u_neighbour_ds(sycl::range<2>(NDRr_L,NDRc_L), h);
          h.parallel_for<class Compute>(nd_range<2>(range<2>(NDRr,NDRc), range<2>(NDRr_L,NDRc_L)), [=](nd_item<2> idx) {
            int i = idx.get_global_id(0);
            int j = idx.get_global_id(1);

            int iLocal = idx.get_local_id(0);
            int jLocal = idx.get_local_id(1);

            int Xi[] = {0, -1,  0,  0,  1};
            int Xj[] = {0,  0, -1,  1,  0};

	          if ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET < r) && (GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET < c))
	          {
	          	double u_neighbour[4];
	          	int neighbour_index = 0;
	          	bool eliminated_cells_ds[5] = { false };

	          	const double u_zero =  calGetMatrixElement(a_d_Sz, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET)
                                     + p_epsilon;
	          	u_neighbour_ds[iLocal][jLocal] = 
                  calGetMatrixElement(a_d_Sz, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET) + 
                  calGetMatrixElement(a_d_Sh, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET);

	          	const double initial_average = calGetMatrixElement(a_d_Sh, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET) - p_epsilon;

	          	for (neighbour_index = 1; neighbour_index < 5; ++neighbour_index)
	          	{
	          		if
	          		(
	          			((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index])       <  (idx.get_group(1)       * NDRr_L + OFFSET))
	          			|| ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index])    >= ((idx.get_group(1) + 1) * NDRr_L + OFFSET))
	          			|| ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) <  (idx.get_group(0)       * NDRr_L + OFFSET))
	          			|| ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) >= ((idx.get_group(0) + 1) * NDRr_L + OFFSET))
	          		)
	          		{
	          			u_neighbour[neighbour_index - 1] =
	          			calGetMatrixElement(a_d_Sz, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index], GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index])
	          				+ calGetMatrixElement(a_d_Sh, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index], GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]);
	          		}
	          	}
	            idx.barrier(sycl::access::fence_space::local_space);

	          	double average = 0.0;
	          	bool again = false;
	          	do
	          	{
	          		again = false;
	          		average = initial_average;
	          		int cell_count = 0;
	          		if (!eliminated_cells_ds[0])
	          		{
	          			average += u_zero;
	          			++cell_count;
	          		}
	          		for (neighbour_index = 1; neighbour_index < 5; ++neighbour_index)
	          		{
	          			if (!eliminated_cells_ds[neighbour_index])
	          			{
	          				if
	          				(
	          					((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) >=       (idx.get_group(1) *      NDRr_L + OFFSET))
	          					&& ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) <    ((idx.get_group(1) + 1) * NDRr_L + OFFSET))
	          					&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) >= (idx.get_group(0) *      NDRr_L + OFFSET))
	          					&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) < ((idx.get_group(0) + 1) * NDRr_L + OFFSET))
	          				)
	          				{
	          					average += u_neighbour_ds[iLocal + Xi[neighbour_index]][jLocal + Xj[neighbour_index]];
	          				}
	          				else
	          				{
	          					average += u_neighbour[neighbour_index - 1];
	          				}
	          				++cell_count;
	          			}
	          		}

	          		if (cell_count != 0)
	          		{
	          			average /= cell_count;
	          		}

	          		if ((!eliminated_cells_ds[0]) && (average <= u_zero))
	          		{
	          			eliminated_cells_ds[0] = true;
	          			again = true;
	          		}
	          		for (neighbour_index = 1; neighbour_index < 5; ++neighbour_index)
	          		{
	          			if (!eliminated_cells_ds[neighbour_index])
	          			{
	          				if
	          				(
	          					((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) >=       (idx.get_group(1)      * NDRr_L + OFFSET))
	          					&& ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) <    ((idx.get_group(1) + 1) * NDRr_L + OFFSET))
	          					&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) >= (idx.get_group(0)      * NDRr_L + OFFSET))
	          					&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) < ((idx.get_group(0) + 1) * NDRr_L + OFFSET))
	          				)
	          				{
	          					if
	          					(
	          						average <= u_neighbour_ds[iLocal + Xi[neighbour_index]][jLocal + Xj[neighbour_index]]
	          					)
	          					{
	          						eliminated_cells_ds[neighbour_index] = true;
	          						again = true;
	          					}
	          				}
	          				else
	          				{
	          					if (average <= u_neighbour[neighbour_index - 1])
	          					{
	          						eliminated_cells_ds[neighbour_index] = true;
	          						again = true;
	          					}
	          				}
	          			}
	          		}
	          	} while (again == true);
	          	for (neighbour_index = 1; neighbour_index < 5; ++neighbour_index)
	          	{
	          		if (!eliminated_cells_ds[neighbour_index])
	          		{
	          			if
	          			(
	          				((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET       + Xi[neighbour_index]) >=(idx.get_group(1)       * NDRr_L + OFFSET))
	          				&& ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET    + Xi[neighbour_index]) < ((idx.get_group(1) + 1) * NDRr_L + OFFSET))
	          				&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) >=(idx.get_group(0)       * NDRr_L + OFFSET))
	          				&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) < ((idx.get_group(0) + 1) * NDRr_L + OFFSET))
	          			)
	          			{
                     calSetBufferedMatrixElement(a_d_Sf, r, c,  (neighbour_index - 1), GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET, (average - u_neighbour_ds[iLocal + Xi[neighbour_index]][jLocal + Xj[neighbour_index]]) * p_r);
	          			}
	          			else
	          			{
                     calSetBufferedMatrixElement(a_d_Sf, r, c, (neighbour_index - 1), GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET, (average - u_neighbour[neighbour_index - 1]) * p_r);
	          			}
	          		}
	          	}
	          }
         });
      });
  
      // balance kernel
      Q.submit([&](handler& h) { 
          auto Sf_ds = accessor<double, 3, access::mode::read_write, access::target::local>(sycl::range<3>(NDRr_L,NDRc_L,4), h);
          auto a_d_Sf = d_Sf->template get_access<access::mode::read>(h);
          auto a_d_Sh = d_Sh->template get_access<access::mode::read_write>(h);

          h.parallel_for<class Update>(nd_range<2>(range<2>(NDRr,NDRc), range<2>(NDRr_L,NDRc_L)), [=](nd_item<2> idx) {
            // int i = idx[1];
            // int j = idx[0];
            int i = idx.get_global_id(0);
            int j = idx.get_global_id(1);

            int iLocal = idx.get_local_id(0);
            int jLocal = idx.get_local_id(1);

            int Xi[] = {0, -1,  0,  0,  1};
            int Xj[] = {0,  0, -1,  1,  0};

            for (int neighbour_index = 0; neighbour_index < ADJACENT_CELLS; ++neighbour_index)
	          {
	          	if ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET < r) && (GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET < c))
	          	{
	          		Sf_ds[iLocal][jLocal][neighbour_index] =
                  calGetBufferedMatrixElement(a_d_Sf, r, c, neighbour_index, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET);
	          	}
	          	else
	          	{
	          		Sf_ds[iLocal][jLocal][neighbour_index] = nodata;
	          	}
	          }
            idx.barrier(sycl::access::fence_space::local_space);

            if
	          ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET < r-1) && (GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET < c-1))
	          {
	          	double h_next = calGetMatrixElement(a_d_Sh, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET);

	          	for (int neighbour_index = 0; neighbour_index < ADJACENT_CELLS; ++neighbour_index)
	          	{
	          		if
	          		(
                  (GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index + 1] >= (      idx.get_group(1)      * NDRr_L + OFFSET))
	          			&& (GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index + 1] < ((   idx.get_group(1) + 1) * NDRr_L + OFFSET))
	          			&& (GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index + 1] >= (idx.get_group(0)      * NDRr_L + OFFSET))
	          			&& (GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index + 1] < ((idx.get_group(0) + 1) * NDRr_L + OFFSET))
	          		)
	          		{
	          			h_next += Sf_ds[iLocal + Xi[neighbour_index + 1]][jLocal + Xj[neighbour_index + 1]][3 - neighbour_index];
	          		}
	          		else
	          		{
	          			h_next += calGetBufferedMatrixElement(a_d_Sf, r, c,  (3 - neighbour_index), GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index + 1], GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index + 1]);
	          		}

	          		h_next -= Sf_ds[iLocal][jLocal][neighbour_index];
	          	}

	          	calSetMatrixElement(a_d_Sh, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET, h_next);
	          }
      	
            });
      });
   }

      Q.wait();
    double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
    printf("Elapsed time: %lf [s]\n", cl_time);

  auto Sh_2 = d_Sh->template get_access<access::mode::read_write>();

  // calSaveMatrix2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]);
  FILE *f;
  f = fopen(argv[OUTPUT_PATH_ID], "w");

  if (!f)
    return false;

    char str[STRLEN];
  int i, j;

  for (i = 0; i < r; i++)
  {
    for (j = 0; j < c; j++)
    {
      sprintf(str, "%f ", calGetMatrixElement(Sh_2, c, i, j));
      fprintf(f, "%s ", str);
    }
    fprintf(f, "\n");
  }

  fclose(f);

   /*** Releasing memory ***/
   printf("Releasing memory...\n");
  //  free(d_Sh, Q);
  //  free(d_Sz, Q);
  //  free(d_Sf, Q);
  //  delete d_Sh;
   delete d_Sz;
   delete d_Sf;
   delete[] Sz;
   delete[] Sh;

  return 0;
}
