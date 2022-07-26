#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "util.hpp"

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5
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
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

#define ROW_STRIDE (gridDim.y * blockDim.y)
#define COLUMN_STRIDE (gridDim.x * blockDim.x)

#define GLOBAL_THREAD_ROW_INDEX (blockIdx.y * blockDim.y + threadIdx.y)
#define GLOBAL_THREAD_COLUMN_INDEX (blockIdx.x * blockDim.x + threadIdx.x)

#define THREADS_PER_BLOCK (32)
#define CELLS_PER_THREAD (1)

#define THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION (16)
#define THREADS_PER_BLOCK_TILING_WIDTH_UPDATE (16)

#define MASK_WIDTH (3)
#define INPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION (THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION + MASK_WIDTH - 1)
#define OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION (THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION)
#define INPUT_TILE_WIDTH_TILING_WIDTH_UPDATE (THREADS_PER_BLOCK_TILING_WIDTH_UPDATE + MASK_WIDTH - 1)
#define OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE (THREADS_PER_BLOCK_TILING_WIDTH_UPDATE)
#define OFFSET 1
// ----------------------------------------------------------------------------
// Neighbourhood definitions
// ----------------------------------------------------------------------------
static __constant__ int Xi[5]; // Xi: von Neuman neighborhood row coordinates (see below)
static __constant__ int Xj[5]; // Xj: von Neuman neighborhood col coordinates (see below)

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
void readHeaderInfo(char* path, int &nrows, int &ncols, /*double &xllcorner, double &yllcorner, double &cellsize,*/ double &nodata)
{
	FILE* f;

	if ( (f = fopen(path,"r") ) == 0){
		printf("%s configuration header file not found\n", path);
		exit(0);
	}

	//Reading the header
	char str[STRLEN];
	fscanf(f,"%s",&str); fscanf(f,"%s",&str); ncols = atoi(str);      //ncols
	fscanf(f,"%s",&str); fscanf(f,"%s",&str); nrows = atoi(str);      //nrows
	fscanf(f,"%s",&str); fscanf(f,"%s",&str); //xllcorner = atof(str);  //xllcorner
	fscanf(f,"%s",&str); fscanf(f,"%s",&str); //yllcorner = atof(str);  //yllcorner
	fscanf(f,"%s",&str); fscanf(f,"%s",&str); //cellsize = atof(str);   //cellsize
	fscanf(f,"%s",&str); fscanf(f,"%s",&str); nodata = atof(str);     //NODATA_value
}

bool loadGrid2D(double *M, int rows, int columns, char *path)
{
	FILE *f = fopen(path, "r");

	if (!f) {
		printf("%s grid file not found\n", path);
		exit(0);
	}

	char str[STRLEN];
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < columns; j++)
		{
			fscanf(f, "%s", str);
			SET(M, columns, i, j, atof(str));
		}

	fclose(f);

	return true;
}

bool saveGrid2Dr(double *M, int rows, int columns, char *path)
{
	FILE *f;
	f = fopen(path, "w");

	if (!f)
		return false;

	char str[STRLEN];
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			sprintf(str, "%f ", GET(M, columns, i, j));
			fprintf(f, "%s ", str);
		}
		fprintf(f, "\n");
	}

	fclose(f);

	return true;
}

double * addLayer2D(int rows, int columns)
{
	double *buffer = NULL;
	const int buffer_size = sizeof(double) * rows * columns;
	cudaError_t error = cudaMallocManaged(&buffer, buffer_size);
	if (error != cudaSuccess)
	{
		return NULL;
	}
	return buffer;
}

void removeLayer2D(double **layer)
{
	if ((layer != NULL) && (*layer != NULL))
	{
		cudaFree(*layer);
		*layer = NULL;
		printf("Released device memory\n");
	}
}

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------

__global__ void sciddicaTSimulationInit
(
	const int i_start, const int j_start,
	const int i_end, const int j_end,
	const int rows, const int columns,
	double* __restrict Sz, double* __restrict Sh
)
{
	int column_index = blockIdx.x * blockDim.x + threadIdx.x;
	int row_index = blockIdx.y * blockDim.y + threadIdx.y;
	if (row_index < rows && column_index < columns) {
			const double fluid_thickness = GET(Sh, columns, row_index, column_index);
			if (fluid_thickness > 0.0)
			{
				const double elevation = GET(Sz, columns, row_index, column_index);
				SET(Sz, columns, row_index, column_index, elevation - fluid_thickness);
			}
		}
	
}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------
__global__ void sciddicaTResetFlows
(
	const int i_start, const int j_start,
	const int i_end, const int j_end,
	const int rows, const int columns,
	const double nodata,
	double* __restrict Sf
)
{
	int column_index = blockIdx.x * blockDim.x + threadIdx.x;
	int row_index = blockIdx.y * blockDim.y + threadIdx.y;
	if (row_index < rows && column_index < columns) {
			// reset the flows to the neighbouring cells
			for (int neighbour_index = 0; neighbour_index < 4; ++neighbour_index)
			{
				BUF_SET(Sf, rows, columns, neighbour_index, row_index, column_index, nodata);
			}
	}
}

__global__ void sciddicaTFlowsComputation
(
	const int i_start, const int j_start,
	const int i_end, const int j_end,
	const int rows, const int columns,
	const double nodata,
	double * __restrict Sz, double * __restrict Sh, double * __restrict Sf,
	const double p_r, const double p_epsilon
)
{
	const int i = blockIdx.y * OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE + threadIdx.y;
	const int j = blockIdx.x * OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE + threadIdx.x;

	if (i < 1 || i > rows-2 || j < 1 || j > columns-2)
	return;


	__shared__ double u_neighbour_ds[INPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION][INPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION];

	bool eliminated_cells[5];
	int neighbour_index = 0;
	double u_zero;
	double initial_average;

	u_neighbour_ds[threadIdx.y+OFFSET  ][threadIdx.x+OFFSET  ] = GET(Sz, columns,   i, j) + GET(Sh, columns, i, j);
	u_neighbour_ds[threadIdx.y+OFFSET-1][threadIdx.x+OFFSET  ] = GET(Sz, columns, i-1, j) + GET(Sh, columns, i-1, j);
	u_neighbour_ds[threadIdx.y+OFFSET+1][threadIdx.x+OFFSET  ] = GET(Sz, columns, i+1, j) + GET(Sh, columns, i+1, j);
	u_neighbour_ds[threadIdx.y+OFFSET  ][threadIdx.x+OFFSET-1] = GET(Sz, columns, i, j-1) + GET(Sh, columns, i, j-1);
	u_neighbour_ds[threadIdx.y+OFFSET  ][threadIdx.x+OFFSET+1] = GET(Sz, columns, i, j+1) + GET(Sh, columns, i, j+1);
	
	u_zero = GET(Sz, columns, i, j) + p_epsilon;
	initial_average = GET(Sh, columns, i, j) - p_epsilon;
	for (neighbour_index = 0; neighbour_index < 5; ++neighbour_index)
	{
		eliminated_cells[neighbour_index] = false;
	}


    double average = 0.0;
    bool again = false;
    do
    {
		again = false;
		average = initial_average;
		int cell_count = 0;

		if (!eliminated_cells[0])
		{
			average += u_zero;
			++cell_count;
		}
		for (neighbour_index = 1; neighbour_index < 5; ++neighbour_index)
		{
			if (!eliminated_cells[neighbour_index])
			{
				average += u_neighbour_ds[threadIdx.y+OFFSET+Xi[neighbour_index]][threadIdx.x+OFFSET+Xj[neighbour_index]];
				++cell_count;
			}
		}

		if (cell_count != 0)
		{
			average /= cell_count;
		}

		if ((!eliminated_cells[0]) && (average <= u_zero))
		{
			eliminated_cells[0] = true;
			again = true;
		}
		for (neighbour_index = 1; neighbour_index < 5; ++neighbour_index)
		{
			if
			(
				(!eliminated_cells[neighbour_index])
				&& (average <= u_neighbour_ds[threadIdx.y+OFFSET+Xi[neighbour_index]][threadIdx.x+OFFSET+Xj[neighbour_index]])
			)
			{
				eliminated_cells[neighbour_index] = true;
				again = true;
			}
		}
	} while (again == true);

	for (neighbour_index = 1; neighbour_index < 5; ++neighbour_index)
	{
		if (!eliminated_cells[neighbour_index])
		{
			BUF_SET(Sf, rows, columns, (neighbour_index - 1), i, j, (average - u_neighbour_ds[threadIdx.y+OFFSET+Xi[neighbour_index]][threadIdx.x+OFFSET+Xj[neighbour_index]]) * p_r);
		}
	}
	
}

__global__ void sciddicaTWidthUpdate
(
	const int i_start, const int j_start,
	const int i_end, const int j_end,
	const int rows, const int columns,
	const double nodata,
	double * __restrict Sz, double * __restrict Sh, double * __restrict Sf
)
{
	const int i = blockIdx.y * OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE + threadIdx.y;
	const int j = blockIdx.x * OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE + threadIdx.x ;

	__shared__ double Sf_ds[INPUT_TILE_WIDTH_TILING_WIDTH_UPDATE][INPUT_TILE_WIDTH_TILING_WIDTH_UPDATE][4];


	if (i < 1 || i > rows-2 || j < 1 || j > columns-2)
	return;

	for (int neighbour_index = 0; neighbour_index < 4; ++neighbour_index)
	{
		Sf_ds[threadIdx.y+OFFSET][threadIdx.x+OFFSET][neighbour_index] =
			BUF_GET(Sf, rows, columns, neighbour_index, i, j);
		Sf_ds[threadIdx.y+OFFSET-1][threadIdx.x+OFFSET][neighbour_index] =
			BUF_GET(Sf, rows, columns, neighbour_index, i-1, j);
		Sf_ds[threadIdx.y+OFFSET+1][threadIdx.x+OFFSET][neighbour_index] =
			BUF_GET(Sf, rows, columns, neighbour_index, i+1, j);
		Sf_ds[threadIdx.y+OFFSET][threadIdx.x+OFFSET+1][neighbour_index] =
			BUF_GET(Sf, rows, columns, neighbour_index, i, j+1);
		Sf_ds[threadIdx.y+OFFSET][threadIdx.x+OFFSET-1][neighbour_index] =
			BUF_GET(Sf, rows, columns, neighbour_index, i, j-1);
	}
	

	const int thread_x = threadIdx.y + OFFSET;
	const int thread_y = threadIdx.x + OFFSET;
	double h_next = GET(Sh, columns, i, j);
	h_next +=
		Sf_ds[thread_x + Xi[1]][thread_y + Xj[1]][3]
		- Sf_ds[thread_x][thread_y][0];
	h_next +=
		Sf_ds[thread_x + Xi[2]][thread_y + Xj[2]][2]
		- Sf_ds[thread_x][thread_y][1];
	h_next +=
		Sf_ds[thread_x + Xi[3]][thread_y + Xj[3]][1]
		- Sf_ds[thread_x][thread_y][2];
	h_next +=
		Sf_ds[thread_x + Xi[4]][thread_y + Xj[4]][0]
		- Sf_ds[thread_x][thread_y][3];
		SET(Sh, columns, i, j, h_next);

}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
	int rows, columns;
	double nodata;
	readHeaderInfo(argv[HEADER_PATH_ID], rows, columns, nodata);

	int i_start = 1, i_end = rows-1;  // [i_start,i_end[: kernels application range along the rows
	int j_start = 1, j_end = columns-1;  // [i_start,i_end[: kernels application range along the rows
	double *Sz;                    // Sz: substate (grid) containing the cells' altitude a.s.l.
	double *Sh;                    // Sh: substate (grid) containing the cells' flow thickness
	double *Sf;                    // Sf: 4 substates containing the flows towards the 4 neighs
	double p_r = P_R;              // p_r: minimization algorithm outflows dumping factor
	double p_epsilon = P_EPSILON;  // p_epsilon: frictional parameter threshold
	int steps = atoi(argv[STEPS_ID]); //steps: simulation steps

	const int Xj_h[] = {0, -1,  0,  0,  1}; // Xi: von Neuman neighborhood row coordinates (see below)
	const int Xi_h[] = {0,  0, -1,  1,  0}; // Xj: von Neuman neighborhood col coordinates (see below)
	cudaMemcpyToSymbol(Xi, &Xi_h, 5 * sizeof(int));
	cudaMemcpyToSymbol(Xj, &Xj_h, 5 * sizeof(int));

	// The adopted von Neuman neighborhood
	// Format: flow_index:cell_label:(row_index,col_index)
	//
	//   cell_label in [0,1,2,3,4]: label assigned to each cell in the neighborhood
	//   flow_index in   [0,1,2,3]: outgoing flow indices in Sf from cell 0 to the others
	//       (row_index,col_index): 2D relative indices of the cells
	//
	//               |0:1:(-1, 0)|
	//   |1:2:( 0,-1)| :0:( 0, 0)|2:3:( 0, 1)|
	//               |3:4:( 1, 0)|
	//
	//

	Sz = addLayer2D(rows, columns);                  // Allocates the Sz substate grid
	Sh = addLayer2D(rows, columns);                  // Allocates the Sh substate grid
	Sf = addLayer2D(ADJACENT_CELLS * rows, columns); // Allocates the Sf substates grid,
																				 //   having one layer for each adjacent cell

	loadGrid2D(Sz, rows, columns, argv[DEM_PATH_ID]);   // Load Sz from file
	loadGrid2D(Sh, rows, columns, argv[SOURCE_PATH_ID]);// Load Sh from file

	// Apply the init kernel (elementary process) to the whole domain grid (cellular space)
	dim3 grid_dimension
	(
		((int) ceil((((float) (i_end - i_start)) / THREADS_PER_BLOCK) / CELLS_PER_THREAD)),
		((int) ceil((((float) (j_end - j_start)) / THREADS_PER_BLOCK) / CELLS_PER_THREAD)),
		1
	);
	dim3 block_dimension(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
	// dim3 grid_dimension(16, 153, 1);
	// dim3 block_dimension(32, 4, 1);

	dim3 grid_dimension_tiling_width_computation
	(
		((int) ceil(((float) (rows)) / OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION)),
		((int) ceil(((float) (columns)) / OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION)),
		1
	);
	dim3 block_dimension_tiling_width_computation(OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION, OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION, 1);

	dim3 grid_dimension_tiling_width_update
	(
		((int) ceil(((float) (rows)) / OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE)),
		((int) ceil(((float) (columns)) / OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE)),
		1
	);
	dim3 block_dimension_tiling_width_update(OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE, OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE, 1);

	sciddicaTSimulationInit<<<grid_dimension, block_dimension>>>(i_start, j_start, i_end, j_end, rows, columns, Sz, Sh);

	util::Timer cl_timer;
	// simulation loop
	for (int step = 0; step < steps; ++step)
	{
		// Apply the resetFlow kernel to the whole domain
		sciddicaTResetFlows<<<grid_dimension, block_dimension>>>(i_start, j_start, i_end, j_end, rows, columns, nodata, Sf);

		sciddicaTFlowsComputation<<<grid_dimension_tiling_width_computation, block_dimension_tiling_width_computation>>>(i_start, j_start, i_end, j_end, rows, columns, nodata, Sz, Sh, Sf, p_r, p_epsilon);

		sciddicaTWidthUpdate<<<grid_dimension_tiling_width_update, block_dimension_tiling_width_update>>>(i_start, j_start, i_end, j_end, rows, columns, nodata, Sz, Sh, Sf);
	}
	cudaDeviceSynchronize();
	double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
	printf("Elapsed time: %lf [s]\n", cl_time);

	saveGrid2Dr(Sh, rows, columns, argv[OUTPUT_PATH_ID]);// Save Sh to file

	printf("Releasing memory...\n");
	removeLayer2D(&Sz);
	removeLayer2D(&Sh);
	removeLayer2D(&Sf);

	return EXIT_SUCCESS;
}
