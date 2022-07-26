#define calSetMatrixElement(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define calGetMatrixElement(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define calGetBufferedMatrixElement(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )
#define calSetBufferedMatrixElement(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )

#define MAX_GROUP_WORK_SIZE 17
#define SHIFT 1
#define ADJACENT_CELLS 4

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


  if (i < 1 || i > r-2 || j < 1 || j > c-2)
      return;
  
  int Xi[] = {0, -1,  0,  0,  1};
  int Xj[] = {0,  0, -1,  1,  0};

  __local double Sf_ds[MAX_GROUP_WORK_SIZE+1][MAX_GROUP_WORK_SIZE+1][ADJACENT_CELLS];

	for (int neighbour_index = 0; neighbour_index < ADJACENT_CELLS; ++neighbour_index)
	{
			Sf_ds[iLocal+SHIFT][jLocal+SHIFT][neighbour_index] = calGetBufferedMatrixElement(Sf, r, c, neighbour_index, i, j);
      Sf_ds[iLocal+SHIFT-1][jLocal+SHIFT][neighbour_index] = calGetBufferedMatrixElement(Sf, r, c, neighbour_index, i-1, j);
      Sf_ds[iLocal+SHIFT+1][jLocal+SHIFT][neighbour_index] = calGetBufferedMatrixElement(Sf, r, c, neighbour_index, i+1, j);
      Sf_ds[iLocal+SHIFT][jLocal+SHIFT+1][neighbour_index] = calGetBufferedMatrixElement(Sf, r, c, neighbour_index, i, j+1);
      Sf_ds[iLocal+SHIFT][jLocal+SHIFT-1][neighbour_index] = calGetBufferedMatrixElement(Sf, r, c, neighbour_index, i, j-1);
	}

	double h_next = calGetMatrixElement(Sh, c, i, j);
	h_next += Sf_ds[iLocal +SHIFT + Xi[1]][jLocal+SHIFT  + Xj[1]][3] - Sf_ds[iLocal+SHIFT ][jLocal+SHIFT ][0];
	h_next += Sf_ds[iLocal +SHIFT + Xi[2]][jLocal+SHIFT  + Xj[2]][2] - Sf_ds[iLocal+SHIFT ][jLocal+SHIFT ][1];
	h_next += Sf_ds[iLocal +SHIFT + Xi[3]][jLocal+SHIFT  + Xj[3]][1] - Sf_ds[iLocal+SHIFT ][jLocal+SHIFT ][2];
	h_next += Sf_ds[iLocal +SHIFT + Xi[4]][jLocal+SHIFT  + Xj[4]][0] - Sf_ds[iLocal+SHIFT ][jLocal+SHIFT ][3];
  calSetMatrixElement(Sh, c, i, j, h_next);

  // int i = get_global_id(1);
  // int j = get_global_id(0);

  // if (i < 1 || i >= r-1 || j < 1 || j >= c-1)
  //     return;

  //if (calGetMatrixElement(Sz, c, i, j) == nodata)
  //    return;

  // int Xi[] = {0, -1,  0,  0,  1};
  // int Xj[] = {0,  0, -1,  1,  0};
  // double h_next;
  // h_next = calGetMatrixElement(Sh, c, i, j);

  // h_next += calGetBufferedMatrixElement(Sf, r, c, 3, i+Xi[1], j+Xj[1]) - calGetBufferedMatrixElement(Sf, r, c, 0, i, j);
  // h_next += calGetBufferedMatrixElement(Sf, r, c, 2, i+Xi[2], j+Xj[2]) - calGetBufferedMatrixElement(Sf, r, c, 1, i, j);
  // h_next += calGetBufferedMatrixElement(Sf, r, c, 1, i+Xi[3], j+Xj[3]) - calGetBufferedMatrixElement(Sf, r, c, 2, i, j);
  // h_next += calGetBufferedMatrixElement(Sf, r, c, 0, i+Xi[4], j+Xj[4]) - calGetBufferedMatrixElement(Sf, r, c, 3, i, j);

  // calSetMatrixElement(Sh, c, i, j, h_next);


}
