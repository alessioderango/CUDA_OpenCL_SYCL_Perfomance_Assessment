#define calSetMatrixElement(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define calGetMatrixElement(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define calGetBufferedMatrixElement(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )
#define calSetBufferedMatrixElement(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )

//#define TILE_WIDTH_TILING_WIDTH 16
#define OFFSET 1
#define GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET (i+OFFSET)
#define GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET (j+OFFSET)
#define ADJACENT_CELLS 4
#define BORDERSIZE 2

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

  int i = get_global_id(1);
  int j = get_global_id(0);

  int iLocal = get_local_id(1);
  int jLocal = get_local_id(0);

  int Xi[] = {0, -1,  0,  0,  1};
  int Xj[] = {0,  0, -1,  1,  0};

	__local double u_neighbour_ds[TILE_WIDTH_TILING_WIDTH][TILE_WIDTH_TILING_WIDTH];

	if ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET < r) && (GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET < c))
	{
		double u_neighbour[4];
		int neighbour_index = 0;
		bool eliminated_cells_ds[5] = { false };

		const double u_zero =  calGetMatrixElement(Sz, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET)
                           + p_epsilon;
		u_neighbour_ds[iLocal][jLocal] = 
        calGetMatrixElement(Sz, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET) + 
        calGetMatrixElement(Sh, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET);

		const double initial_average = calGetMatrixElement(Sh, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET) - p_epsilon;

		for (neighbour_index = 1; neighbour_index < 5; ++neighbour_index)
		{
			if
			(
				((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index])       <  (get_group_id(1)       * TILE_WIDTH_TILING_WIDTH + OFFSET))
				|| ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index])    >= ((get_group_id(1) + 1) * TILE_WIDTH_TILING_WIDTH + OFFSET))
				|| ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) <  (get_group_id(0)       * TILE_WIDTH_TILING_WIDTH + OFFSET))
				|| ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) >= ((get_group_id(0) + 1) * TILE_WIDTH_TILING_WIDTH + OFFSET))
			)
			{
				u_neighbour[neighbour_index - 1] =
				calGetMatrixElement(Sz, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index], GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index])
					+ calGetMatrixElement(Sh, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index], GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]);
			}
		}
	  barrier(CLK_LOCAL_MEM_FENCE);

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
						((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) >=       (get_group_id(1) *      TILE_WIDTH_TILING_WIDTH + OFFSET))
						&& ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) <    ((get_group_id(1) + 1) * TILE_WIDTH_TILING_WIDTH + OFFSET))
						&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) >= (get_group_id(0) *      TILE_WIDTH_TILING_WIDTH + OFFSET))
						&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) < ((get_group_id(0) + 1) * TILE_WIDTH_TILING_WIDTH + OFFSET))
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
						((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) >=       (get_group_id(1)      * TILE_WIDTH_TILING_WIDTH + OFFSET))
						&& ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index]) <    ((get_group_id(1) + 1) * TILE_WIDTH_TILING_WIDTH + OFFSET))
						&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) >= (get_group_id(0)      * TILE_WIDTH_TILING_WIDTH + OFFSET))
						&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) < ((get_group_id(0) + 1) * TILE_WIDTH_TILING_WIDTH + OFFSET))
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
					((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET       + Xi[neighbour_index]) >=(get_group_id(1)       * TILE_WIDTH_TILING_WIDTH + OFFSET))
					&& ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET    + Xi[neighbour_index]) < ((get_group_id(1) + 1) * TILE_WIDTH_TILING_WIDTH + OFFSET))
					&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) >=(get_group_id(0)       * TILE_WIDTH_TILING_WIDTH + OFFSET))
					&& ((GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index]) < ((get_group_id(0) + 1) * TILE_WIDTH_TILING_WIDTH + OFFSET))
				)
				{
           calSetBufferedMatrixElement(Sf, r, c,  (neighbour_index - 1), GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET, (average - u_neighbour_ds[iLocal + Xi[neighbour_index]][jLocal + Xj[neighbour_index]]) * p_r);
				}
				else
				{
           calSetBufferedMatrixElement(Sf, r, c, (neighbour_index - 1), GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET, (average - u_neighbour[neighbour_index - 1]) * p_r);
				}
			}
		}
	}

}
  
