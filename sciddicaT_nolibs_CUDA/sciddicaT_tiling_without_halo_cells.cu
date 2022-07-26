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
#define GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET (GLOBAL_THREAD_ROW_INDEX + i_start)
#define GLOBAL_THREAD_COLUMN_INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET (GLOBAL_THREAD_COLUMN_INDEX + j_start)

//#define THREADS_PER_BLOCK (16)
#define CELLS_PER_THREAD (1)
#define THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION (THREADS_PER_BLOCK)
#define TILE_WIDTH_TILING_WIDTH_COMPUTATION (THREADS_PER_BLOCK)
#define THREADS_PER_BLOCK_TILING_WIDTH_UPDATE (THREADS_PER_BLOCK)

// ----------------------------------------------------------------------------
// Neighbourhood definitions
// ----------------------------------------------------------------------------
static __constant__ int Xi[5];
static __constant__ int Xj[5];

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
			for (int neighbour_index = 0; neighbour_index < ADJACENT_CELLS; ++neighbour_index)
			{
				BUF_SET(Sf, rows, columns, neighbour_index, row_index, column_index, 0.0);
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

	__shared__ double u_neighbour_ds[TILE_WIDTH_TILING_WIDTH_COMPUTATION][TILE_WIDTH_TILING_WIDTH_COMPUTATION];

	if ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET < i_end) && (GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET < j_end))
	{
		double u_neighbour[4];
		int neighbour_index = 0;
		// store which cells were eliminated due to having a larger altitude than the average topographical altitude
		bool eliminated_cells_ds[5] = { false };

		// get the altitude of the current cell
		const double u_zero = GET(Sz, columns, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET) + p_epsilon;
		u_neighbour_ds[threadIdx.y][threadIdx.x] =
			GET(Sz, columns, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET)
			+ GET(Sh, columns, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET);
		const double initial_average = GET(Sh, columns, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET) - p_epsilon;

		for (neighbour_index = 1; neighbour_index < 5; ++neighbour_index)
		{
			if
			(
				((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) < (blockIdx.y * blockDim.y + i_start))
				|| ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) >= ((blockIdx.y + 1) * blockDim.y + i_start))
				|| ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) < (blockIdx.x * blockDim.x + j_start))
				|| ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) >= ((blockIdx.x + 1) * blockDim.x + j_start))
			)
			{
				u_neighbour[neighbour_index - 1] =
					GET(Sz, columns, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index], GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index])
					+ GET(Sh, columns, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index], GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]);
			}
		}
		__syncthreads();

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
						((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) >=       (blockIdx.y *      blockDim.y + i_start))
						&& ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) <    ((blockIdx.y + 1) * blockDim.y + i_start))
						&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) >= (blockIdx.x *      blockDim.x + j_start))
						&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) < ((blockIdx.x + 1) * blockDim.x + j_start))
					)
					{
						average += u_neighbour_ds[threadIdx.y + Xi[neighbour_index]][threadIdx.x + Xj[neighbour_index]];
					}
					else
					{
						average += u_neighbour[neighbour_index - 1];
					}
					++cell_count;
				}
			}

			// calculate the average topographical altitude
			if (cell_count != 0)
			{
				average /= cell_count;
			}

			// check which cells need to be eliminated as their topographical altitude is higher than the average topographical altitude
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
						((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) >=       (blockIdx.y *      blockDim.y + i_start))
						&& ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) <    ((blockIdx.y + 1) * blockDim.y + i_start))
						&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) >= (blockIdx.x *      blockDim.x + j_start))
						&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) < ((blockIdx.x + 1) * blockDim.x + j_start))
					)
					{
						if
						(
							average <= u_neighbour_ds[threadIdx.y + Xi[neighbour_index]][threadIdx.x + Xj[neighbour_index]]
						)
						{
							eliminated_cells_ds[neighbour_index] = true;
							// since at least one cell has a topographical altitude higher than the average one, we need to run another iteration
							again = true;
						}
					}
					else
					{
						if (average <= u_neighbour[neighbour_index - 1])
						{
							eliminated_cells_ds[neighbour_index] = true;
							// since at least one cell has a topographical altitude higher than the average one, we need to run another iteration
							again = true;
						}
					}
				}
			}
		} while (again == true);

		// redistribute the fluid thickness among all cells which have an altitude less than the average altitude in order to achieve the equilibrium criterion
		for (neighbour_index = 1; neighbour_index < 5; ++neighbour_index)
		{
			if (!eliminated_cells_ds[neighbour_index])
			{
				// redistribute the difference between the average topographical altitude and the topographical altitude of the cell reduced by a damping factor
				if
				(
					((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) >=       (blockIdx.y *      blockDim.y + i_start))
					&& ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) < (   (blockIdx.y + 1) * blockDim.y + i_start))
					&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) >= (blockIdx.x *      blockDim.x + j_start))
					&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) < ((blockIdx.x + 1) * blockDim.x + j_start))
				)
				{
					BUF_SET(Sf, rows, columns, (neighbour_index - 1), GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET, (average - u_neighbour_ds[threadIdx.y + Xi[neighbour_index]][threadIdx.x + Xj[neighbour_index]]) * p_r);
				}
				else
				{
					BUF_SET(Sf, rows, columns, (neighbour_index - 1), GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET, (average - u_neighbour[neighbour_index - 1]) * p_r);
				}
			}
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
	__shared__ double Sf_ds[THREADS_PER_BLOCK_TILING_WIDTH_UPDATE][THREADS_PER_BLOCK_TILING_WIDTH_UPDATE][ADJACENT_CELLS];

	#pragma unroll
	for (int neighbour_index = 0; neighbour_index < ADJACENT_CELLS; ++neighbour_index)
	{
		if ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET < i_end) && (GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET < j_end))
		{
			Sf_ds[threadIdx.y][threadIdx.x][neighbour_index] =
				BUF_GET(Sf, rows, columns, neighbour_index, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET);
		}
		else
		{
			Sf_ds[threadIdx.y][threadIdx.x][neighbour_index] = nodata;
		}
	}
	__syncthreads();

	if
	(
		(GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET < i_end)
		&& (GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET < j_end)
	)
	{
		// get the fluid thickness present in the cell in the current step
		double h_next = GET(Sh, columns, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET);
		#pragma unroll
		for (int neighbour_index = 0; neighbour_index < ADJACENT_CELLS; ++neighbour_index)
		{
			if
			(
				(GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index + 1] >= (      blockIdx.y *      blockDim.y + i_start))
				&& (GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index + 1] < ((   blockIdx.y + 1) * blockDim.y + i_start))
				&& (GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index + 1] >= (blockIdx.x *      blockDim.x + j_start))
				&& (GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index + 1] < ((blockIdx.x + 1) * blockDim.x + j_start))
			)
			{
				h_next += Sf_ds[threadIdx.y + Xi[neighbour_index + 1]][threadIdx.x + Xj[neighbour_index + 1]][3 - neighbour_index];
			}
			else
			{
				h_next += BUF_GET(Sf, rows, columns, (3 - neighbour_index), GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index + 1], GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index + 1]);
			}

			h_next -= Sf_ds[threadIdx.y][threadIdx.x][neighbour_index];
		}

		// replace the current fluid thickness with the values considering the current iteration's inflows and outflows to the neighbouring cells
		SET(Sh, columns, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET, h_next);
	}
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
			 ((int) (ceil(((float)columns) / (float)THREADS_PER_BLOCK))),
			 ((int) (ceil(((float)rows) / (float)THREADS_PER_BLOCK))),
			 1
	);
	 dim3 block_dimension(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
	std::cout << "grid dim 0 " <<  grid_dimension.x << std::endl;
	std::cout << "grid dim 1 " <<  grid_dimension.y << std::endl;
	std::cout << "block_dimension dim 0 " <<  THREADS_PER_BLOCK << std::endl;
	std::cout << "block_dimension dim 1 " <<  THREADS_PER_BLOCK << std::endl;
	// dim3 grid_dimension(1, 1, 1);
	// dim3 block_dimension(1, 1, 1);

	dim3 grid_dimension_tiling_width_computation
	(
		((int) ceil(((float) (i_end - i_start)) / THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION)),
		((int) ceil(((float) (j_end - j_start)) / THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION)),
		1
	);
	dim3 block_dimension_tiling_width_computation(min(THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION, (i_end - i_start)), min(THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION, (j_end - j_start)), 1);

	dim3 grid_dimension_tiling_width_update
	(
		((int) ceil(((float) (i_end - i_start)) / THREADS_PER_BLOCK_TILING_WIDTH_UPDATE)),
		((int) ceil(((float) (j_end - j_start)) / THREADS_PER_BLOCK_TILING_WIDTH_UPDATE)),
		1
	);
	dim3 block_dimension_tiling_width_update(THREADS_PER_BLOCK_TILING_WIDTH_UPDATE, THREADS_PER_BLOCK_TILING_WIDTH_UPDATE, 1);

	sciddicaTSimulationInit<<<grid_dimension, block_dimension>>>(i_start, j_start, i_end, j_end, rows, columns, Sz, Sh);

	util::Timer cl_timer;
	// simulation loop
	for (int step = 0; step < steps; ++step)
	{
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
