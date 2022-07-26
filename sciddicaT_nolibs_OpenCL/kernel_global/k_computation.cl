#define calSetMatrixElement(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define calGetMatrixElement(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define calGetBufferedMatrixElement(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )
#define calSetBufferedMatrixElement(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )

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

  if (i < 1 || i >= r-1 || j < 1 || j >= c-1)
      return;

  //if (calGetMatrixElement(Sz, c, i, j) == nodata)
  //    return;

  int Xi[] = {0, -1,  0,  0,  1};
  int Xj[] = {0,  0, -1,  1,  0};

  bool eliminated_cells[5] = {false, false, false, false, false};
  bool again;
  int cells_count;
  double average;
  double m;
  double u[5];
  int n;
  double z, h;

  m = calGetMatrixElement(Sh, c, i, j) - p_epsilon;
  u[0] = calGetMatrixElement(Sz, c, i, j) + p_epsilon;
  z = calGetMatrixElement(Sz, c, i + Xi[1], j + Xj[1]);
  h = calGetMatrixElement(Sh, c, i + Xi[1], j + Xj[1]);
  u[1] = z + h;                                         
  z = calGetMatrixElement(Sz, c, i + Xi[2], j + Xj[2]);
  h = calGetMatrixElement(Sh, c, i + Xi[2], j + Xj[2]);
  u[2] = z + h;                                         
  z = calGetMatrixElement(Sz, c, i + Xi[3], j + Xj[3]);
  h = calGetMatrixElement(Sh, c, i + Xi[3], j + Xj[3]);
  u[3] = z + h;                                         
  z = calGetMatrixElement(Sz, c, i + Xi[4], j + Xj[4]);
  h = calGetMatrixElement(Sh, c, i + Xi[4], j + Xj[4]);
  u[4] = z + h;

  do
  {
    again = false;
    average = m;
    cells_count = 0;

    for (n = 0; n < 5; n++)
      if (!eliminated_cells[n])
      {
        average += u[n];
        cells_count++;
      }

    if (cells_count != 0)
      average /= cells_count;

    for (n = 0; n < 5; n++)
      if ((average <= u[n]) && (!eliminated_cells[n]))
      {
        eliminated_cells[n] = true;
        again = true;
      }
  } while (again);

  if (!eliminated_cells[1]) calSetBufferedMatrixElement(Sf, r, c, 0, i, j, (average - u[1]) * p_r);
  if (!eliminated_cells[2]) calSetBufferedMatrixElement(Sf, r, c, 1, i, j, (average - u[2]) * p_r);
  if (!eliminated_cells[3]) calSetBufferedMatrixElement(Sf, r, c, 2, i, j, (average - u[3]) * p_r);
  if (!eliminated_cells[4]) calSetBufferedMatrixElement(Sf, r, c, 3, i, j, (average - u[4]) * p_r);
}
