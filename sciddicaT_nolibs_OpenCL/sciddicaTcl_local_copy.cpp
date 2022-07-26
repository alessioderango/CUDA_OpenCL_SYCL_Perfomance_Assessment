#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>
#include "cl_err_code.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util.hpp" // utility library

#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5
#define P_R 0.5
#define P_EPSILON 0.001
#define NUMBER_OF_OUTFLOWS 4
#define STRLEN 256

#define calSetMatrixElement(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define calGetMatrixElement(M, columns, i, j) (M[(((i) * (columns)) + (j))])



void readGISInfo(char* path, int &r, int &c, /*double &xllcorner, double &yllcorner, double &cellsize,*/ double &nodata)
{
  FILE* f;
  
  if ( (f = fopen(path,"r") ) == 0){
    printf("Configuration header file not found\n");
    exit(0);
  }

  char str[STRLEN];
  //int cont = -1;
  //fpos_t position;

  //Reading the header
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); c = atoi(str);         //ncols
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); r = atoi(str);         //nrows
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //xllcorner = atof(str); //xllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //yllcorner = atof(str); //yllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //cellsize = atof(str);  //cellsize
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nodata = atof(str);    //NODATA_value 

//  //Checks if actually there are ncols x nrows values into the file
//  fgetpos (f, &position);
//  while (!feof(f))
//  {
//    fscanf(f,"%s",&str);
//    cont++;
//  }
//  fsetpos (f, &position);
//  if (r * c != cont)
//  {
//    printf("File corrupted\n");
//    exit(0);
//  }
}

void calfLoadMatrix2Dr(double *M, int rows, int columns, FILE *f)
{
  char str[STRLEN];
  int i, j;

  for (i = 0; i < rows; i++)
    for (j = 0; j < columns; j++)
    {
      fscanf(f, "%s", str);
      calSetMatrixElement(M, columns, i, j, atof(str));
    }
}

bool calLoadMatrix2Dr(double *M, int rows, int columns, char *path)
{
  FILE *f = NULL;
  f = fopen(path, "r");

  if (!f)
    return false;

  calfLoadMatrix2Dr(M, rows, columns, f);

  fclose(f);

  return true;
}

void calfSaveMatrix2Dr(double *M, int rows, int columns, FILE *f)
{
  char str[STRLEN];
  int i, j;

  for (i = 0; i < rows; i++)
  {
    for (j = 0; j < columns; j++)
    {
      sprintf(str, "%f ", calGetMatrixElement(M, columns, i, j));
      fprintf(f, "%s ", str);
    }
    fprintf(f, "\n");
  }
}

bool calSaveMatrix2Dr(double *M, int rows, int columns, char *path)
{
  FILE *f;
  f = fopen(path, "w");

  if (!f)
    return false;

  calfSaveMatrix2Dr(M, rows, columns, f);

  fclose(f);

  return true;
}

double *calAddSingleLayerSubstate2Dr(int rows, int columns)
{

  double *tmp = (double *)malloc(sizeof(double) * rows * columns);
  if (!tmp)
    return NULL;
  return tmp;
}

void sciddicaTSimulationInit(int r, int c, double* Sz, double* Sh)
{
  double z, h;
  int i, j;

#pragma omp parallel for
  for (i = 0; i < r; i++)
    for (j = 0; j < c; j++)
    {
      h = calGetMatrixElement(Sh, c, i, j);

      if (h > 0.0)
      {
        z = calGetMatrixElement(Sz, c, i, j);
        calSetMatrixElement(Sz, c, i, j, z - h);
      }
    }
}

int main(int argc, char **argv)
{
  int rows, cols;
  double  nodata;
  readGISInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

  double *Sz;
  double *Sh;
  //double *Sf;
  int r = rows;
  int c = cols;
  int i_start = 1, i_end = r-1;
  int j_start = 1, j_end = c-1;
  int Xi[] = {0, -1,  0,  0,  1};
  int Xj[] = {0,  0, -1,  1,  0};
  double p_r = P_R;
  double p_epsilon = P_EPSILON;
  int steps = atoi(argv[STEPS_ID]);
  cl::Buffer d_Sz;
  cl::Buffer d_Sh;
  cl::Buffer d_Sf;

  Sz = calAddSingleLayerSubstate2Dr(r, c);
  Sh = calAddSingleLayerSubstate2Dr(r, c);


  calLoadMatrix2Dr(Sz, r, c, argv[DEM_PATH_ID]);
  calLoadMatrix2Dr(Sh, r, c, argv[SOURCE_PATH_ID]);
  sciddicaTSimulationInit(r, c, Sz, Sh);

  try
  {
    // Discover platforms and devices
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::string s;
    for (int i = 0; i < platforms.size(); i++)
    {
      platforms[i].getInfo(CL_PLATFORM_NAME, &s);
      std::cout << "platform[" << i << "]: " << s << std::endl;
      
      std::vector<cl::Device> devices;
      platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
      for (int j = 0; j < devices.size(); j++ )
      {
        devices[j].getInfo(CL_DEVICE_NAME, &s);
        std::cout << "- device[" << j << "]: " << s << std::endl;
      }
    } 

    int platform_id = 0; //std::cout << "platform = "; std::cin >> platform_id;
    int device_id   = 0; //  std::cout << "device   = "; std::cin >> device_id;
    std::vector<cl::Device> devices;
    platforms[platform_id].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices[device_id];

    // Create a context 
    //cl::Context context(DEVICE); 
    cl::Context context(device);

    // Load in kernel source, creating a program object for the context
    cl::Program p_reset(context, util::loadProgram("./kernel_local_copy/k_reset_local.cl"), true);
    cl::Program p_computation(context, util::loadProgram("./kernel_local_copy/k_computation_local.cl"), true);
    cl::Program p_balance(context, util::loadProgram("./kernel_local_copy/k_balance_local.cl"), true);

    // Get the command queue
    cl::CommandQueue queue(context);

    // Create the kernel functors
    auto k_reset = cl::make_kernel<int, int, int, /*cl::Buffer,*/ cl::Buffer>(p_reset, "k_reset");
    auto k_computation = cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer, double, double>(p_computation, "k_computation");
    auto k_balance = cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer>(p_balance, "k_balance");

    d_Sz = cl::Buffer(context, Sz, Sz+r*c, true);
    d_Sh = cl::Buffer(context, Sh, Sh+r*c, true);
    d_Sf = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * NUMBER_OF_OUTFLOWS*r*c);
    
    queue.finish();
    
    int device_vector_width = 64;
    int NDRr = 608;//device_vector_width * int( 0.5 + (float)r / device_vector_width );
    int NDRc = 512;//device_vector_width * int( 0.5 + (float)c / device_vector_width );
    int NDRr_L = 16;
    int NDRc_L = 16;
    std::cout << "NDRr   = " << NDRr << std::endl;
    std::cout << "NDRc   = "<< NDRc << std::endl;
    std::cout << "NDRr_L   = "<< NDRr_L << std::endl;
    std::cout << "NDRc_L   = "<< NDRc_L << std::endl; 
    cl::NDRange global(NDRr, NDRc);
    cl::NDRange local(NDRr_L, NDRc_L);

    device.getInfo(CL_DEVICE_NAME, &s);
    std::cout << "Running simultaion on the " << s << " OpenCL compliant device..." << std::endl;
    util::Timer cl_timer;
    for (int step = 0; step < steps; step++)
    {
      k_reset( cl::EnqueueArgs( queue, global, local), r, c, nodata, /*d_Sz,*/ d_Sf);
      k_computation( cl::EnqueueArgs( queue, global, local), r, c, nodata, d_Sz, d_Sh, d_Sf, p_epsilon, p_r);
      k_balance( cl::EnqueueArgs( queue, global, local), r, c, nodata, d_Sz, d_Sh, d_Sf);
    }

    queue.finish();

    double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
    printf("Elapsed time: %lf [s]\n", cl_time);

    cl::copy(queue, d_Sh, Sh, Sh+r*c);
    calSaveMatrix2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]);
  }
  catch (cl::Error err) 
  {
    std::cout << "Exception\n";
    std::cerr << "ERROR: " << err.what() << "(" << err_code(err.err()) << ")" << std::endl;
  }

  /*** Releasing memory ***/
  printf("Releasing memory...\n");
  delete[] Sz;
  delete[] Sh;

  return 0;
}
