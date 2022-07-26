//#define calGetMatrixElement(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define calSetBufferedMatrixElement(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )

__kernel void k_reset(
        int r,
        int c,
        int nodata,
        //__global double* Sz,
        __global double* Sf
        )
{
  int i = get_global_id(1);
  int j = get_global_id(0);

  if (i < 1 || i >= r-1 || j < 1 || j >= c-1)
      return;

  calSetBufferedMatrixElement(Sf, r, c, 0, i, j, 0.0);
  calSetBufferedMatrixElement(Sf, r, c, 1, i, j, 0.0);
  calSetBufferedMatrixElement(Sf, r, c, 2, i, j, 0.0);
  calSetBufferedMatrixElement(Sf, r, c, 3, i, j, 0.0);
}
