# definisce la macro CPPC
ifndef CPPC
	CPPC=g++
endif

ifndef CUDACC
	CUDACC=/opt/cuda-10.0/bin/nvcc
endif

#TEST=1
ifdef TEST
	#TEST_B=1

	HDR=../data/test_header.txt
	DEM=../data/test_dem.txt
	ifdef TEST_B
		SRC=../data/test_b_source.txt
	else
		SRC=../data/test_source.txt
	endif
	OUT=./test_output_OpenMP
	OUT_SERIAL=./test_output_serial
	OUT_CUDA=./test_output_cuda
	STEPS=2
else
	HDR=../data/tessina_header.txt
	DEM=../data/tessina_dem.txt
	SRC=../data/tessina_source.txt
	HDR16=../data/tessina_headerX16.txt
	DEM16=../data/tessina_demX16.txt   
	SRC16=../data/tessina_sourceX16.txt
	OUT=./tessina_output_OpenMP
	OUT_SERIAL=./tessina_output_serial
	OUT_CUDA=./tessina_output_cuda
	STEPS=4000 #2680 #2620 #2608 #2621
	STEPS4=32000 
	STEPS16=64000
endif

# definisce le macro contenenti i nomei degli eseguibili
# e il numero di thread omp per la versione parallela
NT = 2 # numero di threads OpenMP
MPI_NODES = 3
EXEC = sciddicaTomp
EXEC_SERIAL = sciddicaTserial
EXEC_CUDA_STRAIGHTFORWARD = sciddicaTcudaStraightforward
EXEC_CUDA_WITHHALOCELLS = sciddicaTcudawithHaloCells
EXEC_CUDA_WITHOUTHALOCELLS = sciddicaTcudaWithoutHaloCells
EXEC_CUDA_WITHHALOCELLS_FULLNEIGHBORHOOD = sciddicaTcudaWithHaloCells_fullneighborhood
EXEC_MPI_CUDA = sciddicaTmpi_cuda
CUDA_COMP_ARCH = -gencode arch=compute_61,code=sm_61
ifeq ($(CUDA_ARCH),AMPERE)
	CUDA_COMP_ARCH = -gencode arch=compute_80,code=sm_80
else
ifeq ($(CUDA_ARCH),TURING)
	CUDA_COMP_ARCH = -gencode arch=compute_75,code=sm_75
else
ifeq ($(CUDA_ARCH),VOLTA)
	CUDA_COMP_ARCH = -gencode arch=compute_70,code=sm_70
else
ifeq ($(CUDA_ARCH),PASCAL)
	CUDA_COMP_ARCH = -gencode arch=compute_61,code=sm_61
else
ifeq ($(CUDA_ARCH),MAXWELL)
CUDA_COMP_ARCH = -gencode arch=compute_52,code=sm_52
else
ifeq ($(CUDA_ARCH),KEPLER)
CUDA_COMP_ARCH = -gencode arch=compute_35,code=sm_35
endif
endif
endif
endif
endif
endif

CUDA_OPT = -O3 -restrict $(CUDA_COMP_ARCH)

# definisce il target di default, utile in
# caso di invocazione di make senza parametri
default:all

# compila le versioni seriale e OpenMP
all: cuda_straightforward cuda_tiling_with_halo_cells cuda_tiling_without_halo_cells 

cuda_straightforward: sciddicaT_straightforward.cu util.hpp
	$(CUDACC) $(CUDA_OPT) sciddicaT_straightforward.cu -o $(EXEC_CUDA_STRAIGHTFORWARD)
 
cuda_tiling_with_halo_cells: sciddicaT_tiling_with_halo_cells.cu util.hpp
	$(CUDACC) $(CUDA_OPT) -DTHREADS_PER_BLOCK=$(TILE_SIZE) sciddicaT_tiling_with_halo_cells.cu -o $(EXEC_CUDA_WITHHALOCELLS)

cuda_tiling_with_halo_cells_fullneighborhood: sciddicaT_tiling_with_halo_cells_fullneighborhood.cu util.hpp
	$(CUDACC) $(CUDA_OPT) sciddicaT_tiling_with_halo_cells_fullneighborhood.cu -o $(EXEC_CUDA_WITHHALOCELLS_FULLNEIGHBORHOOD)

cuda_tiling_without_halo_cells: sciddicaT_tiling_without_halo_cells.cu util.hpp
	$(CUDACC) $(CUDA_OPT) -DTHREADS_PER_BLOCK=$(TILE_SIZE) sciddicaT_tiling_without_halo_cells.cu -o $(EXEC_CUDA_WITHOUTHALOCELLS)

mpi_cuda: sciddicaT_mpi.cu util.hpp
	$(CUDACC) -L/usr/lib/openmpi/ -lmpi sciddicaT_mpi.cu -o $(EXEC_MPI_CUDA)

openmp: sciddicaT.cpp util.hpp
	$(CPPC) sciddicaT.cpp -o $(EXEC) -fopenmp -O3

serial: sciddicaT.cpp util.hpp
	$(CPPC) sciddicaT.cpp -o $(EXEC_SERIAL) -O3

run_straightforward:
	./$(EXEC_CUDA_STRAIGHTFORWARD) $(HDR) $(DEM) $(SRC) $(OUT_CUDA) $(STEPS) $(BLOCK_SIZE_X) $(BLOCK_SIZE_Y) &&  md5sum $(OUT_CUDA) && cat $(HDR) $(OUT_CUDA) > $(OUT_CUDA).qgis && rm $(OUT_CUDA)

run_withouthalocells:
	./$(EXEC_CUDA_WITHOUTHALOCELLS) $(HDR) $(DEM) $(SRC) $(OUT_CUDA) $(STEPS) &&  md5sum $(OUT_CUDA) && cat $(HDR) $(OUT_CUDA) > $(OUT_CUDA).qgis && rm $(OUT_CUDA)

run_withhalocells:
	./$(EXEC_CUDA_WITHHALOCELLS) $(HDR) $(DEM) $(SRC) $(OUT_CUDA) $(STEPS) &&  md5sum $(OUT_CUDA) && cat $(HDR) $(OUT_CUDA) > $(OUT_CUDA).qgis && rm $(OUT_CUDA)

run_sciddicaT_tiling_with_halo_cells_fullneighborhood:
	./$(EXEC_CUDA_WITHHALOCELLS_FULLNEIGHBORHOOD) $(HDR) $(DEM) $(SRC) $(OUT_CUDA) $(STEPS) &&  md5sum $(OUT_CUDA) && cat $(HDR) $(OUT_CUDA) > $(OUT_CUDA).qgis #&& rm $(OUT_CUDA)

run_all:
	./$(EXEC_CUDA_STRAIGHTFORWARD) $(HDR) $(DEM) $(SRC) $(OUT_CUDA) $(STEPS) &&  md5sum $(OUT_CUDA) && cat $(HDR) $(OUT_CUDA) > $(OUT_CUDA).qgis && rm $(OUT_CUDA)
	./$(EXEC_CUDA_WITHHALOCELLS) $(HDR) $(DEM) $(SRC) $(OUT_CUDA) $(STEPS) &&  md5sum $(OUT_CUDA) && cat $(HDR) $(OUT_CUDA) > $(OUT_CUDA).qgis && rm $(OUT_CUDA)
	./$(EXEC_CUDA_WITHOUTHALOCELLS) $(HDR) $(DEM) $(SRC) $(OUT_CUDA) $(STEPS) &&  md5sum $(OUT_CUDA) && cat $(HDR) $(OUT_CUDA) > $(OUT_CUDA).qgis && rm $(OUT_CUDA)

run_straightforward16:
	./$(EXEC_CUDA_STRAIGHTFORWARD) $(HDR16) $(DEM16) $(SRC16) $(OUT_CUDA) $(STEPS16) $(BLOCK_SIZE_X) $(BLOCK_SIZE_Y) &&  md5sum $(OUT_CUDA) && cat $(HDR16) $(OUT_CUDA) > $(OUT_CUDA).qgis && rm $(OUT_CUDA)

run_withouthalocells16:
	./$(EXEC_CUDA_WITHOUTHALOCELLS) $(HDR16) $(DEM16) $(SRC16) $(OUT_CUDA) $(STEPS16) &&  md5sum $(OUT_CUDA) && cat $(HDR16) $(OUT_CUDA) > $(OUT_CUDA).qgis && rm $(OUT_CUDA)

run_withhalocells16:
	./$(EXEC_CUDA_WITHHALOCELLS) $(HDR16) $(DEM16) $(SRC16) $(OUT_CUDA) $(STEPS16) &&  md5sum $(OUT_CUDA) && cat $(HDR16) $(OUT_CUDA) > $(OUT_CUDA).qgis && rm $(OUT_CUDA)

run_all16:
	./$(EXEC_CUDA_STRAIGHTFORWARD) $(HDR16) $(DEM16) $(SRC16) $(OUT_CUDA) $(STEPS16) &&  md5sum $(OUT_CUDA) && cat $(HDR16) $(OUT_CUDA) > $(OUT_CUDA).qgis && rm $(OUT_CUDA)
	./$(EXEC_CUDA_WITHHALOCELLS) $(HDR16) $(DEM16) $(SRC16) $(OUT_CUDA) $(STEPS16) &&  md5sum $(OUT_CUDA) && cat $(HDR16) $(OUT_CUDA) > $(OUT_CUDA).qgis && rm $(OUT_CUDA)
	./$(EXEC_CUDA_WITHOUTHALOCELLS) $(HDR16) $(DEM16) $(SRC16) $(OUT_CUDA) $(STEPS16) &&  md5sum $(OUT_CUDA) && cat $(HDR16) $(OUT_CUDA) > $(OUT_CUDA).qgis && rm $(OUT_CUDA)

run_mpi_cuda:
	mpirun -np $(MPI_NODES) ./$(EXEC_MPI_CUDA) $(HDR) $(DEM) $(SRC) $(OUT_CUDA) $(STEPS) &&  md5sum $(OUT_CUDA) && cat $(HDR) $(OUT_CUDA) > $(OUT_CUDA).qgis #&& rm $(OUT_CUDA)

# esegue la simulazione OpenMP 
run_omp:
	OMP_NUM_THREADS=$(NT) ./$(EXEC) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) &&  md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

# esegue la simulazione seriale
run:
	./$(EXEC_SERIAL) $(HDR) $(DEM) $(SRC) $(OUT_SERIAL) $(STEPS) &&  md5sum $(OUT_SERIAL) && cat $(HDR) $(OUT_SERIAL) > $(OUT_SERIAL).qgis && rm $(OUT_SERIAL)

# elimina l'eseguibile, file oggetto e file di output
clean:
	rm -f $(EXEC) $(EXEC_CUDA_STRAIGHTFORWARD) $(EXEC_CUDA_WITHHALOCELLS) $(EXEC_CUDA_WITHOUTHALOCELLS) $(EXEC_SERIAL) *.o *output*

# elimina file oggetto e di output
wipe:
	rm -f *.o *output*
