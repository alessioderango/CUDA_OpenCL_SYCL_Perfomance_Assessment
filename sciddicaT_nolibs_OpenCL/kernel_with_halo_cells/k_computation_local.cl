#define calSetMatrixElement(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define calGetMatrixElement(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define calGetBufferedMatrixElement(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )
#define calSetBufferedMatrixElement(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )

#define MAX_GROUP_WORK_SIZE 14
#define OFFSET 1
#define BORDERSIZE 2

//#define TILE_SIZE 16
#define THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION (TILE_SIZE-2)
#define THREADS_PER_BLOCK_TILING_WIDTH_UPDATE (TILE_SIZE-2)

#define MASK_WIDTH (3)
#define INPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION (THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION + MASK_WIDTH - 1)
#define OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION (THREADS_PER_BLOCK_TILING_WIDTH_COMPUTATION)
#define INPUT_TILE_WIDTH_TILING_WIDTH_UPDATE (THREADS_PER_BLOCK_TILING_WIDTH_UPDATE + MASK_WIDTH - 1)
#define OUTPUT_TILE_WIDTH_TILING_WIDTH_UPDATE (THREADS_PER_BLOCK_TILING_WIDTH_UPDATE)

__kernel void k_computation(
        int r,
        int c,
        int nodata,
        __global double *Sz,
        __global double *Sh,
        __global double *Sf,
        double p_epsilon,
        double p_r
        )
{
  int i_start = 1, i_end = r-1;
  int j_start = 1, j_end = c-1;

  int i = get_global_id(1);
  int j = get_global_id(0);
  int i_g = get_group_id(1);
  int j_g = get_group_id(0);
  int iLocal = get_local_id(1);
  int jLocal = get_local_id(0);

  // calculate the row index of the cell to calculate the outflows for
  const int row_output = i_g * OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION + iLocal + i_start;
  // calculate the column index of the cell to calculate the outflows for
  const int column_output = j_g * OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION + jLocal + j_start;
  // calculate the row index of the cell to load into the shared memory
  const int row_input = row_output - i_start;
  // calculate the column index of the cell to load into the shared memory
  const int column_input = column_output - j_start;

  local double u_neighbour_ds[INPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION][INPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION];

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
			calGetMatrixElement(Sz, c, row_input, column_input)
			+ calGetMatrixElement(Sh, c, row_input, column_input);
	}
  barrier(CLK_LOCAL_MEM_FENCE);
  
  if
	(
		(iLocal < OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION)
		&& (jLocal < OUTPUT_TILE_WIDTH_TILING_WIDTH_COMPUTATION)
		&& (row_output < i_end) && (column_output < j_end)
	)
	{
		u_zero = calGetMatrixElement(Sz, c, row_output, column_output) + p_epsilon;
		initial_average = calGetMatrixElement(Sh, c, row_output, column_output) - p_epsilon;
		for (neighbour_index = 0; neighbour_index < 5; ++neighbour_index)
		{
			eliminated_cells[neighbour_index] = false;
		}

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
				calSetBufferedMatrixElement(Sf, r, c, (neighbour_index - 1), row_output, column_output, (average - u_neighbour_ds[iLocal + i_start + Xi[neighbour_index]][jLocal + j_start + Xj[neighbour_index]]) * p_r);
			}
		}
	}

}
