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

#define ADJACENT_CELLS 4

#define THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION (TILE_SIZE-2)
#define THREADS_PER_BLOCK_TILING_WIDTH_UPDATE (TILE_SIZE-2)

#define MASK_WIDTH (3)
#define INPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION (THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION + MASK_WIDTH - 1)
#define OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION (THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION)
#define INPUT_TILE_WIDTH_TILING_WIDTH_UPDATE (THREADS_PER_BLOCK_TILING_WIDTH_UPDATE + MASK_WIDTH - 1)
#define OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE (THREADS_PER_BLOCK_TILING_WIDTH_UPDATE)


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
  int NDRr_L = INPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION;
  int NDRc_L = INPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION;

  int NDRr = NDRr_L * ( ceil((float)r / OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE ));
  int NDRc = NDRc_L * ( ceil((float)c / OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE ));
 

  int NDRr_L_reset = TILE_SIZE;
  int NDRc_L_reset = TILE_SIZE;
  int NDRr_reset = NDRr_L_reset * ( ceil((float)r / NDRr_L_reset ));
  int NDRc_reset = NDRc_L_reset * ( ceil((float)c / NDRc_L_reset ));
  for (int step = 0; step < steps; step++)
  {
      // reset kernel
      Q.submit([&](handler& h) {
          auto a_d_Sf = d_Sf->template get_access<access::mode::write>(h);

          h.parallel_for<class Reset>(nd_range<2>(range<2>(NDRr_reset,NDRc_reset), range<2>(NDRr_L_reset,NDRc_L_reset)), [=](nd_item<2> idx) {
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

          auto u_neighbour_ds = accessor<double, 2, access::mode::read_write, access::target::local>(sycl::range<2>(NDRr_L, NDRc_L), h);

          // accessor<double, 2, access::mode::read_write, access::target::local> u_neighbour_ds(sycl::range<2>(NDRr_L,NDRc_L), h);
          h.parallel_for<class Compute>(nd_range<2>(range<2>(NDRr,NDRc), range<2>(NDRr_L, NDRc_L)), [=](nd_item<2> idx) {
            int i = idx.get_global_id(0);
            int j = idx.get_global_id(1);

            int i_g = idx.get_group(0);
            int j_g = idx.get_group(1);

            int iLocal = idx.get_local_id(0);
            int jLocal = idx.get_local_id(1);

          	// calculate the row index of the cell to calculate the outflows for
          	const int row_output = i_g * OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION + iLocal + i_start;
          	// calculate the column index of the cell to calculate the outflows for
          	const int column_output = j_g * OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION + jLocal + j_start;
          	// calculate the row index of the cell to load into the shared memory
          	const int row_input = row_output - i_start;
          	// calculate the column index of the cell to load into the shared memory
          	const int column_input = column_output - j_start;


            int Xi[] = {0, -1,  0,  0,  1};
            int Xj[] = {0,  0, -1,  1,  0};
            int neighbour_index = 0;
            bool eliminated_cells[5];
            double initial_average;
            double u_zero;
	          // check if the indices are inside the boundaries
	          if ((row_input < r) && (column_input < c))
	          {
	          	// load the sum of the topographical altitude and the fluid thickness into the shared memory buffer
	          	u_neighbour_ds[iLocal][jLocal] =
	          		calGetMatrixElement(a_d_Sz, c, row_input, column_input)
	          		+ calGetMatrixElement(a_d_Sh, c, row_input, column_input);
	          }
	          // load several values required so they can be accessed more efficiently during the computation
	          // empirical results have shown that doing this before the synchronisation point increases performance
	          if
	          (
	          	(iLocal < OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION)
	          	&& (jLocal < OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION)
	          	&& (row_output < i_end) && (column_output < j_end)
	          )
	          {
	          	u_zero = calGetMatrixElement(a_d_Sz, c, row_output, column_output) + p_epsilon;
	          	initial_average = calGetMatrixElement(a_d_Sh, c, row_output, column_output) - p_epsilon;
	          	for (neighbour_index = 0; neighbour_index < 5; ++neighbour_index)
	          	{
	          		eliminated_cells[neighbour_index] = false;
	          	}
	          }
	          // synchronise all threads of the block to make sure processing only starts after all data has been loaded into the shared memory
	          idx.barrier(sycl::access::fence_space::local_space);

            // some threads are responsible for only copying a halo cell into the shared memory, while other threads copy an inner cell into the shared memory and calculate the outgoing flows for this cell
	          // the following if statement makes sure that only threads which have copied an inner cell calculate the outflows for this cell
	          if
	          (
	          	(iLocal < OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION)
	          	&& (jLocal < OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION)
	          	&& (row_output < i_end) && (column_output < j_end)
	          )
	          {
	          	// average topographical altitude
	          	double average = 0.0;
	          	// this boolean indicates if another iteration needs to be performed since there has been at least one cell with a topographical altitude higher than the average topographical altitude
	          	bool again = false;
	          	do
	          	{
	          		again = false;
	          		// get the fluid thickness to redistribute from the current central cell
	          		average = initial_average;
	          		// number of non-eliminated cells in the current iteration
	          		int cell_count = 0;

	          		// sum up the topographical altitudes of all non-eliminated cells
	          		// in addition the non-eliminated cells are counted
	          		// first, the central cell is processed, because for this cell only the fluid thickness is added to the average
	          		if (!eliminated_cells[0])
	          		{
	          			average += u_zero;
	          			++cell_count;
	          		}
	          		// include all neighbouring non-eliminated cells in the average calculation
	          		for (neighbour_index = 1; neighbour_index < 5; ++neighbour_index)
	          		{
	          			// check which neighbouring cells have not already been eliminated
	          			if (!eliminated_cells[neighbour_index])
	          			{
	          				// add the neighbouring cell's sum of topographical altitude and fluid thickness to the average
	          				average += u_neighbour_ds[iLocal + i_start + Xi[neighbour_index]][jLocal + j_start + Xj[neighbour_index]];
	          				++cell_count;
	          			}
	          		}

	          		// calculate the average topographical altitude
	          		if (cell_count != 0)
	          		{
	          			average /= cell_count;
	          		}

	          		// check which cells need to be eliminated as their topographical altitude is higher than the average topographical altitude
	          		// again, we start with the central cell as for this cell only the fluid thickness is considered
	          		if ((!eliminated_cells[0]) && (average <= u_zero))
	          		{
	          			// mark the central cell as eliminated
	          			eliminated_cells[0] = true;
	          			// since at least one cell has a topographical altitude higher than the average one, we need to run another iteration
	          			again = true;
	          		}
	          		// process the neighbouring cells
	          		for (neighbour_index = 1; neighbour_index < 5; ++neighbour_index)
	          		{
	          			// check if the currently iterated neighbouring has not been eliminated and has a topographical altitude greater than the average topographical altitude
	          			if
	          			(
	          				(!eliminated_cells[neighbour_index])
	          				&& (average <= u_neighbour_ds[iLocal + i_start + Xi[neighbour_index]][jLocal + j_start + Xj[neighbour_index]])
	          			)
	          			{
	          				// mark the currently iterated neighbouring cell as eliminated
	          				eliminated_cells[neighbour_index] = true;
	          				// since at least one cell has a topographical altitude higher than the average one, we need to run another iteration
	          				again = true;
	          			}
	          		}
	          	} while (again == true);

	          	// redistribute the fluid thickness among all cells which have an altitude less than the average altitude in order to achieve the equilibrium criterion
	          	for (neighbour_index = 1; neighbour_index < 5; ++neighbour_index)
	          	{
	          		// check if the cell has not already been eliminated
	          		if (!eliminated_cells[neighbour_index])
	          		{
	          			// redistribute the difference between the average topographical altitude and the topographical altitude of the cell reduced by a damping factor
	          			calSetBufferedMatrixElement(a_d_Sf, r, c, (neighbour_index - 1), row_output, column_output, (average - u_neighbour_ds[iLocal + i_start + Xi[neighbour_index]][jLocal + j_start + Xj[neighbour_index]]) * p_r);
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

            int i_g = idx.get_group(0);
            int j_g = idx.get_group(1);

            int iLocal = idx.get_local_id(0);
            int jLocal = idx.get_local_id(1);

         	  // calculate the row index of the cell to calculate the outflows for
          	const int row_output = i_g * OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION + iLocal + i_start;
          	// calculate the column index of the cell to calculate the outflows for
          	const int column_output = j_g * OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION + jLocal + j_start;
          	// calculate the row index of the cell to load into the shared memory
          	const int row_input = row_output - i_start;
          	// calculate the column index of the cell to load into the shared memory
          	const int column_input = column_output - j_start;

            int Xi[] = {0, -1,  0,  0,  1};
            int Xj[] = {0,  0, -1,  1,  0};

	          if ((row_input < r) && (column_input < c))
	          {
	          	// copy the cells representing the current cell's outflows in all four directions into shared memory
	          	for (int neighbour_index = 0; neighbour_index < 4; ++neighbour_index)
	          	{
	          		Sf_ds[iLocal][jLocal][neighbour_index] =
	          			calGetBufferedMatrixElement(a_d_Sf, r, c, neighbour_index, row_input, column_input);
	          	}
	          }
            idx.barrier(sycl::access::fence_space::local_space);
            
            // some threads are responsible for only copying a halo cell into the shared memory, while other threads copy an inner cell into the shared memory and update the fluid thickness for this cell
	          // the following if statement makes sure that only threads which have copied an inner cell updat the cell's fluid thickness
	          if
	          (
	          	(jLocal < OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE)
	          	&& (iLocal < OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE)
	          	&& (row_output < i_end) && (column_output < j_end)
	          )
	          {
	          	// a little shortcut in order to have to type less
	          	const int thread_x = iLocal + i_start;
	          	const int thread_y = jLocal + j_start;
	          	// get the fluid thickness present in the cell in the current step
	          	double h_next = calGetMatrixElement(a_d_Sh, c, row_output, column_output);

	          	h_next +=
	          		// add the incoming flow from the north neighbouring cell
	          		Sf_ds[thread_x + Xi[1]][thread_y + Xj[1]][3]
	          		// subtract the outgoing flow to the north neighbouring cell
	          		- Sf_ds[thread_x][thread_y][0];
	          	h_next +=
	          		// add the incoming flow from the east neighbouring cell
	          		Sf_ds[thread_x + Xi[2]][thread_y + Xj[2]][2]
	          		// subtract the outgoing flow to the east neighbouring cell
	          		- Sf_ds[thread_x][thread_y][1];
	          	h_next +=
	          		// add the incoming flow from the west neighbouring cell
	          		Sf_ds[thread_x + Xi[3]][thread_y + Xj[3]][1]
	          		// subtract the outgoing flow to the west neighbouring cell
	          		- Sf_ds[thread_x][thread_y][2];
	          	h_next +=
	          		// add the incoming flow from the south neighbouring cell
	          		Sf_ds[thread_x + Xi[4]][thread_y + Xj[4]][0]
	          		// subtract the outgoing flow to the south neighbouring cell
	          		- Sf_ds[thread_x][thread_y][3];

	          	// replace the current fluid thickness with the values considering the current iteration's inflows and outflows to the neighbouring cells
	          	calSetMatrixElement(a_d_Sh, c, row_output, column_output, h_next);
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
