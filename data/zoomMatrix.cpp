#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
using namespace std;


void calfLoadMatrix2Dr(int dimY, int dimX,int zoom){

  FILE* source = fopen ("./tessina_source.txt", "r"); 
  FILE* dem = fopen("./tessina_dem.txt", "r");

  char str_H[999];
  char str_Z[999];

  double **Msource = new double*[dimY*zoom];
  for (int i=0; i<dimY*zoom; i++)     
  {
    Msource[i] = new double[dimX*zoom];
  }

  double **Mdem = new double*[dimY*zoom];
  for (int i=0; i<dimY*zoom; i++)     
  {
    Mdem[i] = new double[dimX*zoom];
  }

  for (int y=0; y<dimY; y++)     
    for (int x=0; x<dimX; x++){
      double h;
      double z;

      fscanf(source, "%s", str_H);
      fscanf(dem, "%s", str_Z);
      //printf("%s\n", str_H);
      //                        printf("%d %d\n", y,x);
      h = atof(str_H);
      z = atof(str_Z);
      //                        if(h > 0)
      //                                printf("BBB\n");
      for(int i = 0; i < zoom; i++)
      {
        for(int j = 0; j < zoom; j++)
        {

          Msource[y*zoom+i][x*zoom+j] = h;
          Mdem[y*zoom+i][x*zoom+j] = z;
        }
      }


    }


  //Parameter.setMaxH(max
  fclose(source);
  fclose(dem);

  std::string pathsource = "./tessina_sourceX"+ to_string(zoom*zoom)+".txt";
  std::string pathdem = "./tessina_demX"+  to_string(zoom*zoom)+".txt";

  FILE* sourceX2 = fopen (pathsource.c_str(), "w");
  FILE* demX2 = fopen(pathdem.c_str(), "w");
  for (int y=0; y<dimY*zoom; y++)     
  {
    for (int x=0; x<dimX*zoom; x++){
      double h =  Msource[y][x];
      double z =  Mdem[y][x];
      //                if(h > 0)
      //                   printf("AAA\n");
      fprintf(sourceX2, "%f\t", h);
      fprintf(demX2, "%f\t", z);

    }
    fprintf(sourceX2, "\n");
    fprintf(demX2, "\n");
  }



  fclose(sourceX2);
  fclose(demX2);

}

int main(int argc, char *argv[])
{
  if(argc ==1)
  {printf("inserisci il valore di zoom \n");
    exit(0);
  }int zoom = atoi(argv[1]);

  calfLoadMatrix2Dr(610,496,zoom);



}
