#define calSetMatrixElement(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define calGetMatrixElement(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define calGetBufferedMatrixElement(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )
#define calSetBufferedMatrixElement(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )

//#define THREADS_PER_BLOCK_TILING_WIDTH_UPDATE 16
#define OFFSET 1
#define GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET (i+OFFSET)
#define GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET (j+OFFSET)
#define ADJACENT_CELLS 4
#define BORDERSIZE 2

__kernel void k_balance(
        int r,
        int c,
        int nodata,
        __global double * Sz,
        __global double * Sh,
        __global double * Sf)
{
  int i = get_global_id(1);
  int j = get_global_id(0);

  int iLocal = get_local_id(1);
  int jLocal = get_local_id(0);

  int Xi[] = {0, -1,  0,  0,  1};
  int Xj[] = {0,  0, -1,  1,  0};

  __local double Sf_ds[THREADS_PER_BLOCK_TILING_WIDTH_UPDATE][THREADS_PER_BLOCK_TILING_WIDTH_UPDATE][ADJACENT_CELLS];

	for (int neighbour_index = 0; neighbour_index < ADJACENT_CELLS; ++neighbour_index)
	{
		if ((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET < r) && (GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET < c))
		{
			Sf_ds[iLocal][jLocal][neighbour_index] =
        calGetBufferedMatrixElement(Sf, r, c, neighbour_index, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET);
		}
		else
		{
			Sf_ds[iLocal][jLocal][neighbour_index] = nodata;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if
	((GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET < r-1) && (GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET < c-1))
	{
		double h_next = calGetMatrixElement(Sh, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET);

		for (int neighbour_index = 0; neighbour_index < ADJACENT_CELLS; ++neighbour_index)
		{
			if
			(
        (GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index + 1] >= (      get_group_id(1)      * THREADS_PER_BLOCK_TILING_WIDTH_UPDATE + OFFSET))
				&& (GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index + 1] < ((   get_group_id(1) + 1) * THREADS_PER_BLOCK_TILING_WIDTH_UPDATE + OFFSET))
				&& (GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index + 1] >= (get_group_id(0)      * THREADS_PER_BLOCK_TILING_WIDTH_UPDATE + OFFSET))
				&& (GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index + 1] < ((get_group_id(0) + 1) * THREADS_PER_BLOCK_TILING_WIDTH_UPDATE + OFFSET))
			)
			{
				h_next += Sf_ds[iLocal + Xi[neighbour_index + 1]][jLocal + Xj[neighbour_index + 1]][3 - neighbour_index];
			}
			else
			{
				h_next += calGetBufferedMatrixElement(Sf, r, c,  (3 - neighbour_index), GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET + Xi[neighbour_index + 1], GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET + Xj[neighbour_index + 1]);
			}

			h_next -= Sf_ds[iLocal][jLocal][neighbour_index];
		}

		calSetMatrixElement(Sh, c, GLOBAL_THREAD_ROW_INDEX_WITH_OFFSET, GLOBAL_THREAD_COLUMN_INDEX_WITH_OFFSET, h_next);
	}

}
