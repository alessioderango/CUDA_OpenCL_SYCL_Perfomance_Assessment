#include "util.hpp"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5
#define BLOCK_SIZE_X 6
#define BLOCK_SIZE_Y 7
// ----------------------------------------------------------------------------
// Simulation parameters
// ----------------------------------------------------------------------------
#define P_R 0.5
#define P_EPSILON 0.001
#define ADJACENT_CELLS 4
#define STRLEN 256

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value)                              \
  ((M)[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))] = (value))
#define BUF_GET(M, rows, columns, n, i, j)                                     \
  (M[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))])

#define ROW_STRIDE (gridDim.x * blockDim.x)
#define COLUMN_STRIDE (gridDim.y * blockDim.y)

// #define THREADS_PER_BLOCK (16)
// #define THREADS_PER_BLOCK_X (16)
// #define THREADS_PER_BLOCK_Y (16)
// #define CELLS_PER_THREAD (1)

// ----------------------------------------------------------------------------
// Neighbourhood definitions
// ----------------------------------------------------------------------------
static __constant__ int
    Xi[5]; // Xi: von Neuman neighborhood row coordinates (see below)
static __constant__ int
    Xj[5]; // Xj: von Neuman neighborhood col coordinates (see below)

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
void readHeaderInfo(char *path, int &nrows, int &ncols,
                    /*double &xllcorner, double &yllcorner, double &cellsize,*/
                    double &nodata) {
  FILE *f;

  if ((f = fopen(path, "r")) == 0) {
    printf("%s configuration header file not found\n", path);
    exit(0);
  }

  // Reading the header
  char str[STRLEN];
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  ncols = atoi(str); // ncols
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  nrows = atoi(str); // nrows
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str); // xllcorner = atof(str);  //xllcorner
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str); // yllcorner = atof(str);  //yllcorner
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str); // cellsize = atof(str);   //cellsize
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  nodata = atof(str); // NODATA_value
}

bool loadGrid2D(double *M, int rows, int columns, char *path) {
  FILE *f = fopen(path, "r");

  if (!f) {
    printf("%s grid file not found\n", path);
    exit(0);
  }

  char str[STRLEN];
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < columns; j++) {
      fscanf(f, "%s", str);
      SET(M, columns, i, j, atof(str));
    }

  fclose(f);

  return true;
}

bool saveGrid2Dr(double *M, int rows, int columns, char *path) {
  FILE *f;
  f = fopen(path, "w");

  if (!f)
    return false;

  char str[STRLEN];
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      sprintf(str, "%f ", GET(M, columns, i, j));
      fprintf(f, "%s ", str);
    }
    fprintf(f, "\n");
  }

  fclose(f);

  return true;
}

double *addLayer2D(const int rows, const int columns) {
  double *buffer = (double *)malloc(sizeof(double) * rows * columns);
  return buffer;
}

double *addLayer2D_device(const int rows, const int columns) {
  double *buffer = NULL;
  const int buffer_size = sizeof(double) * rows * columns;
  cudaError_t error = cudaMalloc((void **)&buffer, buffer_size);
  if (error != cudaSuccess) {
    return NULL;
  }
  return buffer;
}

void removeLayer2D(double **layer) {
  if ((layer != NULL) && (*layer != NULL)) {
    free(*layer);
    *layer = NULL;
    printf("Released host memory\n");
  }
}

void removeLayer2D_device(double **layer) {
  if ((layer != NULL) && (*layer != NULL)) {
    cudaFree(*layer);
    *layer = NULL;
    printf("Released device memory\n");
  }
}

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------

__global__ void sciddicaTSimulationInit(const int i_start, const int j_start,
                                        const int i_end, const int j_end,
                                        const int rows, const int columns,
                                        double *__restrict Sz,
                                        double *__restrict Sh) {
  // for (int row_index = blockIdx.x * blockDim.x + threadIdx.x + i_start;
  //      row_index < i_end; row_index += ROW_STRIDE) {
  //   for (int column_index = blockIdx.y * blockDim.y + threadIdx.y + j_start;
  //        column_index < j_end; column_index += COLUMN_STRIDE) {
  int column_index = blockIdx.x * blockDim.x + threadIdx.x;
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
      if (row_index < rows && column_index < columns) {
        //printf(" %d \n", column_index+row_index*columns);
        const double fluid_thickness =
            GET(Sh, columns, row_index, column_index);
        if (fluid_thickness > 0.0) {
          const double elevation = GET(Sz, columns, row_index, column_index);
          SET(Sz, columns, row_index, column_index,
              elevation - fluid_thickness);
        }
      }
  //   }
  // }
}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------
__global__ void sciddicaTResetFlows(const int i_start, const int j_start,
                                    const int i_end, const int j_end,
                                    const int rows, const int columns,
                                    const double nodata,
                                    double *__restrict Sf) {
  /*for (int row_index = blockIdx.x * blockDim.x + threadIdx.x; row_index <
  rows; row_index += ROW_STRIDE)
  {
          for (int column_index = blockIdx.y * blockDim.y + threadIdx.y;
  column_index < columns; column_index += COLUMN_STRIDE)
          {
                  // reset the flows to the neighbouring cells
                  for (int neighbour_index = 0; neighbour_index <
  ADJACENT_CELLS; ++neighbour_index)
                  {
                          BUF_SET(Sf, rows, columns, neighbour_index, row_index,
  column_index, 0.0);
                  }
          }
  }*/
  int column_index = blockIdx.x * blockDim.x + threadIdx.x;
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  if (row_index < rows && column_index < columns) {
  
    BUF_SET(Sf, rows, columns, 0, row_index, column_index, 0.0);
    BUF_SET(Sf, rows, columns, 1, row_index, column_index, 0.0);
    BUF_SET(Sf, rows, columns, 2, row_index, column_index, 0.0);
    BUF_SET(Sf, rows, columns, 3, row_index, column_index, 0.0);
  }
}

__global__ void
sciddicaTFlowsComputation(const int i_start, const int j_start, const int i_end,
                          const int j_end, const int rows, const int columns,
                          const double nodata, double *__restrict Sz,
                          double *__restrict Sh, double *__restrict Sf,
                          const double p_r, const double p_epsilon) {
  int column_index = blockIdx.x * blockDim.x + threadIdx.x;
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  if (row_index < rows-1 && column_index < columns-1 && row_index > 0 && column_index > 0 ) {

  // for (int row_index = blockIdx.x * blockDim.x + threadIdx.x + i_start;
  // row_index < i_end; row_index += ROW_STRIDE)
  // {
  // 	for (int column_index = blockIdx.y * blockDim.y + threadIdx.y + j_start;
  // column_index < j_end; column_index += COLUMN_STRIDE)
  // 	{
  double u[5] = {0.0};

  // get the altitude of the current cell
  u[0] = GET(Sz, columns, row_index, column_index) + p_epsilon;

  // get the altitude and the fluid thickness of the neighbouring cells
  // int neighbour_index = 0;
  // for (neighbour_index = 1; neighbour_index < 5; ++neighbour_index)
  // {
  // 	u[neighbour_index] =
  // 		// get the altitude of the currently iterated neighbouring cell
  // 		GET(Sz, columns, row_index + Xi[neighbour_index], column_index +
  // Xj[neighbour_index])
  // 		// get the fluid thickness of the currently iterated neighbouring
  // cell
  // 		+ GET(Sh, columns, row_index + Xi[neighbour_index], column_index +
  // Xj[neighbour_index]);
  // }

  u[1] = GET(Sz, columns, row_index + Xi[1], column_index + Xj[1]) +
         GET(Sh, columns, row_index + Xi[1], column_index + Xj[1]);
  u[2] = GET(Sz, columns, row_index + Xi[2], column_index + Xj[2]) +
         GET(Sh, columns, row_index + Xi[2], column_index + Xj[2]);
  u[3] = GET(Sz, columns, row_index + Xi[3], column_index + Xj[3]) +
         GET(Sh, columns, row_index + Xi[3], column_index + Xj[3]);
  u[4] = GET(Sz, columns, row_index + Xi[4], column_index + Xj[4]) +
         GET(Sh, columns, row_index + Xi[4], column_index + Xj[4]);

  // store which cells were eliminated due to having a larger altitude than the
  // average topographical altitude
  bool eliminated_cells[5] = {false, false, false, false, false};
  // average topographical altitude
  double average = 0.0;
  // this boolean indicates if another iteration needs to be performed since
  // there has been at least one cell with a topographical altitude higher than
  // the average topographical altitude
  bool again = false;
 
  
  do {
    again = false;
    // get the fluid thickness to redistribute from the current central cell
    average = GET(Sh, columns, row_index, column_index) - p_epsilon;
    // number of non-eliminated cells in the current iteration
    short cell_count = 0;

    // sum up the topographical altitudes of all non-eliminated cells
    // in addition the non-eliminated cells are counted
    for (int neighbour_index = 0; neighbour_index < 5; ++neighbour_index) {
      if (!eliminated_cells[neighbour_index]) {
        average += u[neighbour_index];
        ++cell_count;
      }
    }

    // calculate the average topographical altitude
    if (cell_count != 0) {
      average /= cell_count;
    }

    // check which cells need to be eliminated as their topographical altitude
    // is higher than the average topographical altitude
    for (int neighbour_index = 0; neighbour_index < 5; ++neighbour_index) {
      if ((!eliminated_cells[neighbour_index]) &&
          (average <= u[neighbour_index])) {
        eliminated_cells[neighbour_index] = true;
        // since at least one cell has a topographical altitude higher than the
        // average one, we need to run another iteration
        again = true;
      }
    }
  } while (again == true);
  


  // redistribute the fluid thickness among all cells which have an altitude
  // less than the average altitude in order to achieve the equilibrium
  // criterion for (int neighbour_index = 1; neighbour_index < 5;
  // ++neighbour_index)
  // {
  // 	if (!eliminated_cells[neighbour_index])
  // 	{
  // 		// redistribute the difference between the average topographical
  // altitude and the topographical altitude of the cell reduced by a damping
  // factor 		BUF_SET(Sf, rows, columns, (neighbour_index - 1), row_index,
  // column_index, (average - u[neighbour_index]) * p_r);
  // 	}
  // }

  if (!eliminated_cells[1])
    BUF_SET(Sf, rows, columns, (0), row_index, column_index,
            (average - u[1]) * p_r);
  if (!eliminated_cells[2])
    BUF_SET(Sf, rows, columns, (1), row_index, column_index,
            (average - u[2]) * p_r);
  if (!eliminated_cells[3])
    BUF_SET(Sf, rows, columns, (2), row_index, column_index,
            (average - u[3]) * p_r);
  if (!eliminated_cells[4])
    BUF_SET(Sf, rows, columns, (3), row_index, column_index,
            (average - u[4]) * p_r);
  // 	}
  // }
  }
}

__global__ void sciddicaTWidthUpdate(const int i_start, const int j_start,
                                     const int i_end, const int j_end,
                                     const int rows, const int columns,
                                     const double nodata, double *__restrict Sz,
                                     double *__restrict Sh,
                                     double *__restrict Sf) {
  int column_index = blockIdx.x * blockDim.x + threadIdx.x;
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  if (row_index < rows && column_index < columns) {
  // for (int row_index = blockIdx.x * blockDim.x + threadIdx.x + i_start;
  // row_index < i_end; row_index += ROW_STRIDE)
  // {
  // 	for (int column_index = blockIdx.y * blockDim.y + threadIdx.y + j_start;
  // column_index < j_end; column_index += COLUMN_STRIDE)
  // 	{
  // get the fluid thickness present in the cell in the current step
  double h_next = GET(Sh, columns, row_index, column_index);

  h_next +=
      // add the incoming flow from the north neighbouring cell
      BUF_GET(Sf, rows, columns, 3, row_index + Xi[1], column_index + Xj[1])
      // subtract the outgoing flow to the north neighbouring cell
      - BUF_GET(Sf, rows, columns, 0, row_index, column_index);
  h_next +=
      // add the incoming flow from the east neighbouring cell
      BUF_GET(Sf, rows, columns, 2, row_index + Xi[2], column_index + Xj[2])
      // subtract the outgoing flow to the east neighbouring cell
      - BUF_GET(Sf, rows, columns, 1, row_index, column_index);
  h_next +=
      // add the incoming flow from the west neighbouring cell
      BUF_GET(Sf, rows, columns, 1, row_index + Xi[3], column_index + Xj[3])
      // subtract the outgoing flow to the west neighbouring cell
      - BUF_GET(Sf, rows, columns, 2, row_index, column_index);
  h_next +=
      // add the incoming flow from the south neighbouring cell
      BUF_GET(Sf, rows, columns, 0, row_index + Xi[4], column_index + Xj[4])
      // subtract the outgoing flow to the south neighbouring cell
      - BUF_GET(Sf, rows, columns, 3, row_index, column_index);

  // replace the current fluid thickness with the values considering the current
  // iteration's inflows and outflows to the neighbouring cells
  SET(Sh, columns, row_index, column_index, h_next);
  // 	}
  // }
  }
}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv) {
  int rows, columns;
  double nodata;
  readHeaderInfo(argv[HEADER_PATH_ID], rows, columns, nodata);

  int THREADS_PER_BLOCK_X = atoi(argv[BLOCK_SIZE_X]);
  int THREADS_PER_BLOCK_Y = atoi(argv[BLOCK_SIZE_Y]);

  int i_start = 1,
      i_end =
          rows - 1; // [i_start,i_end[: kernels application range along the rows
  int j_start = 1,
      j_end = columns -
              1;    // [i_start,i_end[: kernels application range along the rows
  double *Sz;       // Sz: substate (grid) containing the cells' altitude a.s.l.
  double *Sh;       // Sh: substate (grid) containing the cells' flow thickness
  double *Sf;       // Sf: 4 substates containing the flows towards the 4 neighs
  double p_r = P_R; // p_r: minimization algorithm outflows dumping factor
  double p_epsilon = P_EPSILON;     // p_epsilon: frictional parameter threshold
  int steps = atoi(argv[STEPS_ID]); // steps: simulation steps

  const int Xj_h[] = {
      0, -1, 0, 0,
      1}; // Xi: von Neuman neighborhood row coordinates (see below)
  const int Xi_h[] = {
      0, 0, -1, 1,
      0}; // Xj: von Neuman neighborhood col coordinates (see below)
  cudaMemcpyToSymbol(Xi, &Xi_h, 5 * sizeof(int));
  cudaMemcpyToSymbol(Xj, &Xj_h, 5 * sizeof(int));

  // The adopted von Neuman neighborhood
  // Format: flow_index:cell_label:(row_index,col_index)
  //
  //   cell_label in [0,1,2,3,4]: label assigned to each cell in the
  //   neighborhood flow_index in   [0,1,2,3]: outgoing flow indices in Sf from
  //   cell 0 to the others
  //       (row_index,col_index): 2D relative indices of the cells
  //
  //               |0:1:(-1, 0)|
  //   |1:2:( 0,-1)| :0:( 0, 0)|2:3:( 0, 1)|
  //               |3:4:( 1, 0)|
  //
  //

  Sz = addLayer2D(rows, columns); // Allocates the Sz substate grid
  Sh = addLayer2D(rows, columns); // Allocates the Sh substate grid
  Sf = addLayer2D(ADJACENT_CELLS * rows,
                  columns); // Allocates the Sf substates grid,
                            //   having one layer for each adjacent cell

  loadGrid2D(Sz, rows, columns, argv[DEM_PATH_ID]);    // Load Sz from file
  loadGrid2D(Sh, rows, columns, argv[SOURCE_PATH_ID]); // Load Sh from file

  double *d_Sz = NULL;
  double *d_Sh = NULL;
  double *d_Sf = NULL;
  d_Sz = addLayer2D_device(rows, columns);
  d_Sh = addLayer2D_device(rows, columns);
  d_Sf = addLayer2D_device(ADJACENT_CELLS * rows, columns);
  cudaMemcpy(d_Sz, Sz, rows * columns * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sh, Sh, rows * columns * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sf, Sf, ADJACENT_CELLS * rows * columns * sizeof(double),
             cudaMemcpyHostToDevice);

  // Apply the init kernel (elementary process) to the whole domain grid
  // (cellular space)
  /* dim3 grid_dimension
  (
           ((int) ceil((((float) (j_end - j_start)) / THREADS_PER_BLOCK_X) /
   CELLS_PER_THREAD)),
           ((int) ceil((((float) (i_end - i_start)) / THREADS_PER_BLOCK_Y) /
   CELLS_PER_THREAD)),
           1
  );
  dim3 block_dimension(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);*/
  dim3 grid_dimension
  (
           (ceil((((float)columns) / (float)THREADS_PER_BLOCK_X))),
           (ceil((((float)rows) / (float)THREADS_PER_BLOCK_Y))),
           1
  );
   dim3 block_dimension(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
  std::cout << "grid dim 0 " <<  grid_dimension.x << std::endl;
  std::cout << "grid dim 1 " <<  grid_dimension.y << std::endl;
  std::cout << "block_dimension dim 0 " <<  THREADS_PER_BLOCK_X << std::endl;
  std::cout << "block_dimension dim 1 " <<  THREADS_PER_BLOCK_Y << std::endl;
  std::cout << "steps " <<  steps << std::endl;
  //dim3 grid_dimension(16, 153, 1);
  //dim3 block_dimension(32, 4, 1);
  // dim3 grid_dimension(1, 1, 1);
  // dim3 block_dimension(1, 1, 1);
  sciddicaTSimulationInit<<<grid_dimension, block_dimension>>>(
      i_start, j_start, i_end, j_end, rows, columns, d_Sz, d_Sh);

  util::Timer cl_timer;
  // simulation loop
  for (int step = 0; step < steps; ++step) {
    // Apply the resetFlow kernel to the whole domain
    sciddicaTResetFlows<<<grid_dimension, block_dimension>>>(
        i_start, j_start, i_end, j_end, rows, columns, nodata, d_Sf);
    /*
    cudaMemcpy(Sf, d_Sf, ADJACENT_CELLS * rows * columns * sizeof(double),
    cudaMemcpyDeviceToHost); for (int i = 0; i < rows; ++i) { for (int j = 0; j
    < columns; ++j) { for (int direction = 0; direction < ADJACENT_CELLS;
    ++direction) { printf("%.3f ", BUF_GET(Sf, rows, columns, direction, i, j));
                    }
                    printf("|| ");
            }
            printf("\n");
    }
    printf("---\n");
    */

    sciddicaTFlowsComputation<<<grid_dimension, block_dimension>>>(
        i_start, j_start, i_end, j_end, rows, columns, nodata, d_Sz, d_Sh, d_Sf,
        p_r, p_epsilon);

    sciddicaTWidthUpdate<<<grid_dimension, block_dimension>>>(
        i_start, j_start, i_end, j_end, rows, columns, nodata, d_Sz, d_Sh,
        d_Sf);
    /*
    cudaMemcpy(Sh, d_Sh, rows * columns * sizeof(double),
    cudaMemcpyDeviceToHost); printf("Sh after step %d:\n", step); for (int i =
    0; i < rows; ++i) { for (int j = 0; j < columns; ++j) { printf("%.2f ", Sh[i
    * columns + j]);
            }
            printf("\n");
    }
    printf("\n");
    */
  }
  cudaDeviceSynchronize();
  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  cudaMemcpy(Sh, d_Sh, rows * columns * sizeof(double), cudaMemcpyDeviceToHost);
  printf("Elapsed time: %lf [s]\n", cl_time);

  saveGrid2Dr(Sh, rows, columns, argv[OUTPUT_PATH_ID]); // Save Sh to file

  printf("Releasing memory...\n");
  removeLayer2D(&Sz);
  removeLayer2D(&Sh);
  removeLayer2D(&Sf);
  removeLayer2D_device(&d_Sz);
  removeLayer2D_device(&d_Sh);
  removeLayer2D_device(&d_Sf);

  return EXIT_SUCCESS;
}
