#define calSetMatrixElement(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define calGetMatrixElement(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define calGetBufferedMatrixElement(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )
#define calSetBufferedMatrixElement(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )

#define MAX_GROUP_WORK_SIZE 14
#define OFFSET 1
#define ADJACENT_CELLS 4
#define BORDERSIZE 2

//#define TILE_SIZE 16
#define THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION (TILE_SIZE-2)
#define THREADS_PER_BLOCK_TILING_WIDTH_UPDATE (TILE_SIZE-2)

#define MASK_WIDTH (3)
#define INPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION (THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION + MASK_WIDTH - 1)
#define OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION (THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION)
#define INPUT_TILE_WIDTH_TILING_WIDTH_UPDATE (THREADS_PER_BLOCK_TILING_WIDTH_UPDATE + MASK_WIDTH - 1)
#define OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE (THREADS_PER_BLOCK_TILING_WIDTH_UPDATE)

__kernel void k_balance(
        int r,
        int c,
        int nodata,
        __global double * Sz,
        __global double * Sh,
        __global double * Sf)
{
  int i_start = 1, i_end = r-1;
  int j_start = 1, j_end = c-1;

  int i = get_global_id(1);
  int j = get_global_id(0);
  int i_g = get_group_id(1);
  int j_g = get_group_id(0);
  int iLocal = get_local_id(1);
  int jLocal = get_local_id(0);
  int Xi[] = {0, -1,  0,  0,  1};
  int Xj[] = {0,  0, -1,  1,  0};

  // calculate the row index of the cell to calculate the outflows for
  const int row_output = i_g * OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION + iLocal + i_start;
  // calculate the column index of the cell to calculate the outflows for
  const int column_output = j_g * OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION + jLocal + j_start;
  // calculate the row index of the cell to load into the shared memory
  const int row_input = row_output - i_start;
  // calculate the column index of the cell to load into the shared memory
  const int column_input = column_output - j_start;

  local double Sf_ds[INPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION][INPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION][4];

	if ((row_input < r) && (column_input < c))
	{
		// copy the cells representing the current cell's outflows in all four directions into shared memory
		for (int neighbour_index = 0; neighbour_index < 4; ++neighbour_index)
		{
			Sf_ds[iLocal][jLocal][neighbour_index] =
				calGetBufferedMatrixElement(Sf, r, c, neighbour_index, row_input, column_input);
		}
	}
  barrier(CLK_LOCAL_MEM_FENCE);
            
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
		double h_next = calGetMatrixElement(Sh, c, row_output, column_output);
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
		calSetMatrixElement(Sh, c, row_output, column_output, h_next);
	}
}
