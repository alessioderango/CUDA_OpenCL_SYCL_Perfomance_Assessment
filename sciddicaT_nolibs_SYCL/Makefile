# definisce la macro CPPC
CPPC_DPCPP=clang++ 
FLAGS_DPCPP=  -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_61 --cuda-path=/opt/cuda

ifeq ($(CUDA_ARCH),AMPERE)
	CUDA_COMP_ARCH = sm_80
else
ifeq ($(CUDA_ARCH),TURING)
	CUDA_COMP_ARCH = sm_75
else
ifeq ($(CUDA_ARCH),VOLTA)
	CUDA_COMP_ARCH = sm_70
else
ifeq ($(CUDA_ARCH),PASCAL)
	CUDA_COMP_ARCH = sm_61
else
ifeq ($(CUDA_ARCH),MAXWELL)
  CUDA_COMP_ARCH = sm_52
else
ifeq ($(CUDA_ARCH),KEPLER)
  CUDA_COMP_ARCH = sm_35
endif
endif
endif
endif
endif
endif
 
CPPC_HIPSYCL=syclcc
FLAGS_HIPSYCL= -Wno-everything  -O3 --hipsycl-platform=cuda --cuda-path=/opt/cuda -L/opt/cuda/lib64 --hipsycl-gpu-arch=$(CUDA_COMP_ARCH) 

HDR=../data/tessina_header.txt
DEM=../data/tessina_dem.txt
SRC=../data/tessina_source.txt
HDR16=../data/tessina_headerX16.txt
DEM16=../data/tessina_demX16.txt   
SRC16=../data/tessina_sourceX16.txt
OUT=./tessina_output_OpenCL
STEPS=4000
STEPS16=64000

# definisce la macro EXEC al nome dell'eseguibile
EXEC_HIPSYCL = sciddicaTsycl_hipsycl
EXEC_HIPSYCL_COL_MAJOR = sciddicaTsycl_hipsycl_col_major
EXEC_DPCPP = sciddicaTsycl_dpcpp
EXEC_DPCPP_COL_MAJOR = sciddicaTsycl_dpcpp_cols_major
EXEC_DPCPP_WHITHOUT_HALO_CELLS = sciddicaTsycl_dpcpp_without_halo_cells
EXEC_HIPSYCL_WHITHOUT_HALO_CELLS = sciddicaTsycl_hipsycl_without_halo_cells
EXEC_DPCPP_WITH_HALO_CELLS =  sciddicaTsycl_dpcpp_with_halo_cells

# definisce il target di default a $(EXEC)
# in caso di invocazione di make senza parametri
default: $(EXEC_HIPSYCL) $(EXEC_DPCPP) $(EXEC_DPCPP_WHITHOUT_HALO_CELLS) $(EXEC_HIPSYCL_WHITHOUT_HALO_CELLS)

#default: $(EXEC_DPCPP) $(EXEC_DPCPP_WHITHOUT_HALO_CELLS) 

hipsycl: $(EXEC_HIPSYCL)
dpcpp: $(EXEC_DPCPP)

# esegue il linkaggio dei file oggetto
# generando l'eseguibile; 
# $@ è una macro predefinita di make che 
# corrisponde a ciò che compare prima del 
# simbolo : ($(EXEC) nel caso specifico);
# S^ è la macro predefinita che corrisponde
# a ciò che compare dopo il simbolo :
# (sciddicaTcl.o nel caso specifico)

$(EXEC_HIPSYCL): sciddicaTsycl_hipsycl.o
	$(CPPC_HIPSYCL) $(FLAGS_HIPSYCL) sciddicaTsycl.o -o $@

$(EXEC_HIPSYCL_COL_MAJOR): sciddicaTsycl_hipsycl_cols_major.o
	$(CPPC_HIPSYCL) $(FLAGS_HIPSYCL) sciddicaTsycl_cols_major.o -o $@

# genera il file oggetto del sorgente cpp
sciddicaTsycl_hipsycl_cols_major.o: sciddicaTsycl_cols_major.cpp 
	$(CPPC_HIPSYCL) $(FLAGS_HIPSYCL) -DHIPSYCL $^ -c

$(EXEC_DPCPP): sciddicaTsycl_dpcpp.o
	$(CPPC_DPCPP) $(FLAGS_DPCPP) sciddicaTsycl.o -o $@

sciddicaTsycl_dpcpp.o: sciddicaTsycl.cpp 
	$(CPPC_DPCPP) $(FLAGS_DPCPP) -DDPCPP $^ -c 

$(EXEC_DPCPP_COL_MAJOR): sciddicaTsycl_dpcpp_cols_major.o
	$(CPPC_DPCPP) $(FLAGS_DPCPP) sciddicaTsycl_cols_major.o -o $@

sciddicaTsycl_dpcpp_cols_major.o: sciddicaTsycl_cols_major.cpp 
	$(CPPC_DPCPP) $(FLAGS_DPCPP) -DDPCPP $^ -c 

# genera il file oggetto del sorgente cpp
sciddicaTsycl_hipsycl.o: sciddicaTsycl.cpp 
	$(CPPC_HIPSYCL) $(FLAGS_HIPSYCL) -DHIPSYCL $^ -c 

$(EXEC_DPCPP_WHITHOUT_HALO_CELLS): 
	$(CPPC_DPCPP) $(FLAGS_DPCPP) sciddicaTsycl_without_halo_cells.cpp -DDPCPP -o $@

$(EXEC_DPCPP_WITH_HALO_CELLS): 
	$(CPPC_DPCPP) $(FLAGS_DPCPP) sciddicaTsycl_with_halo_cells.cpp -DDPCPP -o $@

$(EXEC_HIPSYCL_WHITHOUT_HALO_CELLS): 
	$(CPPC_HIPSYCL) $(FLAGS_HIPSYCL) sciddicaTsycl_without_halo_cells.cpp -DHIPSYCL -o $@


# esegue la simulazione
run_straightforward_hipsycl:
	$(RUN) ./$(EXEC_HIPSYCL) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) $(BLOCK_SIZE_X) $(BLOCK_SIZE_Y) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

run_straightforward_hipsycl_col_major:
	$(RUN) ./$(EXEC_HIPSYCL_COL_MAJOR) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) $(BLOCK_SIZE_X) $(BLOCK_SIZE_Y) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)


run_hipsycl_without_halo_cells:
	$(RUN) ./$(EXEC_HIPSYCL_WHITHOUT_HALO_CELLS) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) $(TILE_SIZE) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

run_straightforward_dpcpp:
	$(RUN) ./$(EXEC_DPCPP) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) $(BLOCK_SIZE_X) $(BLOCK_SIZE_Y) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

run_straightforward_dpcpp_cols_major:
	$(RUN) ./$(EXEC_DPCPP_COL_MAJOR) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) $(BLOCK_SIZE_X) $(BLOCK_SIZE_Y) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)


run_dpcpp_without_halo_cells:
	$(RUN) ./$(EXEC_DPCPP_WHITHOUT_HALO_CELLS) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) $(TILE_SIZE) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

run_dpcpp_with_halo_cells:
	$(RUN) ./$(EXEC_DPCPP_WITH_HALO_CELLS) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) $(TILE_SIZE) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)


run_all:
	$(RUN) ./$(EXEC_HIPSYCL) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)
	$(RUN) ./$(EXEC_DPCPP) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)
	$(RUN) ./$(EXEC_HIPSYCL_WHITHOUT_HALO_CELLS) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)
	$(RUN) ./$(EXEC_DPCPP_WHITHOUT_HALO_CELLS) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

run_all16:
	$(RUN) ./$(EXEC_HIPSYCL) $(HDR16) $(DEM16) $(SRC16) $(OUT) $(STEPS16) && md5sum $(OUT) && cat $(HDR16) $(OUT) > $(OUT).qgis && rm $(OUT)
	$(RUN) ./$(EXEC_DPCPP) $(HDR16) $(DEM16) $(SRC16) $(OUT) $(STEPS16) && md5sum $(OUT) && cat $(HDR16) $(OUT) > $(OUT).qgis && rm $(OUT)
	$(RUN) ./$(EXEC_HIPSYCL_WHITHOUT_HALO_CELLS) $(HDR16) $(DEM16) $(SRC16) $(OUT) $(STEPS16) && md5sum $(OUT) && cat $(HDR16) $(OUT) > $(OUT).qgis && rm $(OUT)
	$(RUN) ./$(EXEC_DPCPP_WHITHOUT_HALO_CELLS) $(HDR16) $(DEM16) $(SRC16) $(OUT) $(STEPS16) && md5sum $(OUT) && cat $(HDR16) $(OUT) > $(OUT).qgis && rm $(OUT)

run_straightforward_hipsycl16:
	$(RUN) ./$(EXEC_HIPSYCL) $(HDR16) $(DEM16) $(SRC16) $(OUT) $(STEPS16)  $(BLOCK_SIZE_X) $(BLOCK_SIZE_Y) && md5sum $(OUT) && cat $(HDR16) $(OUT) > $(OUT).qgis && rm $(OUT)
run_straightforward_dpcpp16:
	$(RUN) ./$(EXEC_DPCPP) $(HDR16) $(DEM16) $(SRC16) $(OUT) $(STEPS16)  $(BLOCK_SIZE_X) $(BLOCK_SIZE_Y) && md5sum $(OUT) && cat $(HDR16) $(OUT) > $(OUT).qgis && rm $(OUT)
run_withhout_halo_cells_hipsycl16:
	$(RUN) ./$(EXEC_HIPSYCL_WHITHOUT_HALO_CELLS) $(HDR16) $(DEM16) $(SRC16) $(OUT) $(STEPS16) $(TILE_SIZE) && md5sum $(OUT) && cat $(HDR16) $(OUT) > $(OUT).qgis && rm $(OUT)
run_withhout_halo_cells_dpcpp16:
	$(RUN) ./$(EXEC_DPCPP_WHITHOUT_HALO_CELLS) $(HDR16) $(DEM16) $(SRC16) $(OUT) $(STEPS16) $(TILE_SIZE) && md5sum $(OUT) && cat $(HDR16) $(OUT) > $(OUT).qgis && rm $(OUT)
run_with_halo_cells_dpcpp16:
	$(RUN) ./$(EXEC_DPCPP_WITH_HALO_CELLS) $(HDR16) $(DEM16) $(SRC16) $(OUT) $(STEPS16) $(TILE_SIZE) && md5sum $(OUT) && cat $(HDR16) $(OUT) > $(OUT).qgis && rm $(OUT)


# elimina l'eseguibile, file oggetto e file di output
clean:
	rm -f $(EXEC_HIPSYCL) $(EXEC_DPCPP) $(EXEC_DPCPP_WHITHOUT_HALO_CELLS) $(EXEC_DPCPP_WITH_HALO_CELLS) $(EXEC_HIPSYCL_WHITHOUT_HALO_CELLS) *.o *output*

# elimina file oggetto e di output
wipe:
	rm -f *.o *output*
