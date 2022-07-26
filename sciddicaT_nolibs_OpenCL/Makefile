# definisce la macro CPPC
ifndef CPPC
	CPPC=g++
endif

# ifeq ($(HOST),JPDM1)
# 	OPENCL_INCLUDE_PATH=-I/opt/cuda-10.0/include/
# 	OPENCL_LIB_PATH=-L/opt/cuda-10.0/lib64/
# else
# 	OPENCL_INCLUDE_PATH=-I/opt/cuda/targets/x86_64-linux/include/
# 	OPENCL_LIB_PATH=-L/opt/cuda/targets/x86_64-linux/lib/
# endif
OPENCL_INCLUDE_PATH=-I$(OPENCL_INC)
OPENCL_LIB_PATH=-L$(OPENCL_LIB)

ifdef TEST
	HDR=../data/test_header.txt
	DEM=../data/test_dem.txt
	SRC=../data/test_source.txt
	OUT=./test_output_OpenCL
	STEPS=2
else
	HDR=../data/tessina_header.txt
	DEM=../data/tessina_dem.txt   
	SRC=../data/tessina_source.txt
	OUT=./tessina_output_OpenCL
	HDR16=../data/tessina_headerX16.txt
	DEM16=../data/tessina_demX16.txt   
	SRC16=../data/tessina_sourceX16.txt
	STEPS=4000
	STEPS4=16000
	STEPS16=64000
endif

# definisce la macro LIBS 
LIBS = -lOpenCL
FLAGS = -O3 -w
# definisce la macro EXEC al nome dell'eseguibile
EXEC = sciddicaTcl
EXEC_WITH_HALO_CELLS = sciddicaTcl_with_halo_cells
EXEC_LOCAL_COPY = sciddicaTcl_local_copy
EXEC_WITHOUT_HALO_CELLS = sciddicaTcl_without_halo_cells

# definisce il target di default a $(EXEC)
# in caso di invocazione di make senza parametri
default: $(EXEC) $(EXEC_WITHOUT_HALO_CELLS) 

# esegue il linkaggio dei file oggetto
# generando l'eseguibile; 
# $@ è una macro predefinita di make che 
# corrisponde a ciò che compare prima del 
# simbolo : ($(EXEC) nel caso specifico);
# S^ è la macro predefinita che corrisponde
# a ciò che compare dopo il simbolo :
# (sciddicaTcl.o nel caso specifico)
$(EXEC): sciddicaTcl.o
	$(CPPC) $^ $(OPENCL_LIB_PATH) $(LIBS) -o $@

# genera il file oggetto del sorgente cpp
sciddicaTcl.o: sciddicaTcl.cpp 
	$(CPPC) $^ $(FLAGS) -c $(OPENCL_INCLUDE_PATH) 

$(EXEC_WITH_HALO_CELLS): sciddicaTcl_with_halo_cells.o
	$(CPPC) $^ $(OPENCL_LIB_PATH) $(LIBS) -o $@

# genera il file oggetto del sorgente cpp
sciddicaTcl_with_halo_cells.o: sciddicaTcl_with_halo_cells.cpp 
	$(CPPC) $^ $(FLAGS) -c $(OPENCL_INCLUDE_PATH) 

$(EXEC_LOCAL_COPY): sciddicaTcl_local_copy.o
	$(CPPC) $^ $(OPENCL_LIB_PATH) $(LIBS) -o $@

# genera il file oggetto del sorgente cpp
sciddicaTcl_local_copy.o: sciddicaTcl_local_copy.cpp 
	$(CPPC) $^ $(FLAGS) -c $(OPENCL_INCLUDE_PATH) 

$(EXEC_WITHOUT_HALO_CELLS): sciddicaTcl_without_halo_cells.o
	$(CPPC) $^ $(OPENCL_LIB_PATH) $(LIBS) -o $@

# genera il file oggetto del sorgente cpp
sciddicaTcl_without_halo_cells.o: sciddicaTcl_without_halo_cells.cpp 
	$(CPPC) $^ $(FLAGS) -c $(OPENCL_INCLUDE_PATH) 

# esegue la simulazione
run_straightforward:
	$(RUN) ./$(EXEC) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) $(BLOCK_SIZE_X) $(BLOCK_SIZE_Y) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

run_with_halo_cells:
	$(RUN) ./$(EXEC_WITH_HALO_CELLS) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) $(TILE_SIZE) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

run_local_copy:
	$(RUN) ./$(EXEC_LOCAL_COPY) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

run_without_halo_cells:
	$(RUN) ./$(EXEC_WITHOUT_HALO_CELLS) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) $(TILE_SIZE) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

run_all:
	$(RUN) ./$(EXEC) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)
	$(RUN) ./$(EXEC_WITHOUT_HALO_CELLS) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)
	$(RUN) ./$(EXEC_WITH_HALO_CELLS) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

run_all16:
	$(RUN) ./$(EXEC) $(HDR16) $(DEM16) $(SRC16) $(OUT) $(STEPS16) && md5sum $(OUT) && cat $(HDR16) $(OUT) > $(OUT).qgis && rm $(OUT)
	$(RUN) ./$(EXEC_WITHOUT_HALO_CELLS) $(HDR16) $(DEM16) $(SRC16) $(OUT) $(STEPS16) && md5sum $(OUT) && cat $(HDR16) $(OUT) > $(OUT).qgis && rm $(OUT)
	$(RUN) ./$(EXEC_WITH_HALO_CELLS) $(HDR16) $(DEM16) $(SRC16) $(OUT) $(STEPS16) && md5sum $(OUT) && cat $(HDR16) $(OUT) > $(OUT).qgis && rm $(OUT)

run_straightforward16:
	$(RUN) ./$(EXEC) $(HDR16) $(DEM16) $(SRC16) $(OUT) $(STEPS16) $(BLOCK_SIZE_X) $(BLOCK_SIZE_Y) && md5sum $(OUT) && cat $(HDR16) $(OUT) > $(OUT).qgis && rm $(OUT)
run_without_halo_cells16:
	$(RUN) ./$(EXEC_WITHOUT_HALO_CELLS) $(HDR16) $(DEM16) $(SRC16) $(OUT) $(STEPS16) $(TILE_SIZE) && md5sum $(OUT) && cat $(HDR16) $(OUT) > $(OUT).qgis && rm $(OUT)
run_with_halo_cells16:
	$(RUN) ./$(EXEC_WITH_HALO_CELLS) $(HDR16) $(DEM16) $(SRC16) $(OUT) $(STEPS16) $(TILE_SIZE) && md5sum $(OUT) && cat $(HDR16) $(OUT) > $(OUT).qgis && rm $(OUT)

# elimina l'eseguibile, file oggetto e file di output
clean:
	rm -f $(EXEC) $(EXEC_WITH_HALO_CELLS) $(EXEC_WITHOUT_HALO_CELLS) *.o *output*

# elimina file oggetto e di output
wipe:
	rm -f *.o *output*
