#define calSetMatrixElement(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define calGetMatrixElement(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define calGetBufferedMatrixElement(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )
#define calSetBufferedMatrixElement(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )

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

  if (i < 1 || i >= r-1 || j < 1 || j >= c-1)
      return;

  //if (calGetMatrixElement(Sz, c, i, j) == nodata)
  //    return;

  int Xi[] = {0, -1,  0,  0,  1};
  int Xj[] = {0,  0, -1,  1,  0};

  double h_next;
  h_next = calGetMatrixElement(Sh, c, i, j);
  h_next += calGetBufferedMatrixElement(Sf, r, c, 3, i+Xi[1], j+Xj[1]) - calGetBufferedMatrixElement(Sf, r, c, 0, i, j);
  h_next += calGetBufferedMatrixElement(Sf, r, c, 2, i+Xi[2], j+Xj[2]) - calGetBufferedMatrixElement(Sf, r, c, 1, i, j);
  h_next += calGetBufferedMatrixElement(Sf, r, c, 1, i+Xi[3], j+Xj[3]) - calGetBufferedMatrixElement(Sf, r, c, 2, i, j);
  h_next += calGetBufferedMatrixElement(Sf, r, c, 0, i+Xi[4], j+Xj[4]) - calGetBufferedMatrixElement(Sf, r, c, 3, i, j);

  calSetMatrixElement(Sh, c, i, j, h_next);
}
