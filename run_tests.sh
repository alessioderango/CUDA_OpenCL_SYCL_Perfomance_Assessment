#!/bin/sh

if [ "$1" != "-gpu" ] || [ "$2" == "" ]
then
  echo ""
  echo "Usage: sh -gpu <gpu_id>"
  echo ""
  echo "Examples:"
  echo "  test on gpu 0:"
  echo "    sh $0 -gpu 0"
  echo ""
  echo "  test on gpu 1:"
  echo "    sh $0 -gpu 1"
  echo ""
  exit 0
fi

HOSTNAME=`echo $HOSTNAME`
echo "HOSTNAME $HOSTNAME"
source ./setenv_$HOSTNAME.txt
export CUDA_VISIBLE_DEVICES=$2

#OutputDirRoot=./LogOuput_
OutputDirRoot=./LogOuput_`date +"%Y-%m-%d_%T"`
mkdir -p ${OutputDirRoot}

OutputFileName=${OutputDirRoot}/results.csv
OutputFileNameLocal=${OutputDirRoot}/results_local.csv
numberoftests=1

declare -a tile_size=( 4 8 16 32 )
declare -a block_size_x=( 4 8 16 32 )
declare -a block_size_y=( 4 8 16 32 )

# SYCL_HIPSYCL                 
# ./sciddicaT_nolibs_SYCL      
# ${OutputDirRoot}/SYCL_HIPSYCL
# sciddicaTsycl_hipsycl        
# run_straightforward_hipsycl  
# run_straightforward_hipsycl_ 
# ".."                         

# SYCL_HIPSYCL_WITHOUT_HALO_CELLS                  
# ./sciddicaT_nolibs_SYCL                          
# ${OutputDirRoot}/SYCL_HIPSYCL_WITHOUT_HALO_CELLS 
# sciddicaTsycl_hipsycl_without_halo_cells         
# run_hipsycl_without_halo_cells                   
# run_run_hipsycl_without_halo_cells_              

declare -a NameCSVOUTPUT=(       CUDA                           OpenCL                        SYCL_DPCCC                    )  
declare -a FolderTest=(          ./sciddicaT_nolibs_CUDA/       ./sciddicaT_nolibs_OpenCL     ./sciddicaT_nolibs_SYCL       )
declare -a OutputDir=(           ${OutputDirRoot}/CUDA          ${OutputDirRoot}/OpenCL       ${OutputDirRoot}/SYCL_DPCPP   )
declare -a TargetMake=(          cuda_straightforward           sciddicaTcl                   sciddicaTsycl_dpcpp           )
declare -a TargetRunMake=(       run_straightforward            run_straightforward           run_straightforward_dpcpp     )
declare -a NameTest=(            run_straightforward_cuda_      run_straightforward_opencl    run_straightforward_dpcpp_    )
declare -a homeFolder=(          ".."                           ".."                          ".."                          )

declare -a NameCSVOUTPUTLocal=(  CUDA_Tiled_withhalocells                    CUDA_Tiled_withouthalocells                    OpenCL_without_halocells                   OpenCL_with_halocells                     SYCL_DPCCC_WITHOUT_HALO_CELLS                   SYCL_DPCCC_WITH_HALO_CELLS                     )
declare -a FolderTestLocal=(     ./sciddicaT_nolibs_CUDA/                    ./sciddicaT_nolibs_CUDA/                       ./sciddicaT_nolibs_OpenCL                  ./sciddicaT_nolibs_OpenCL                 ./sciddicaT_nolibs_SYCL                         ./sciddicaT_nolibs_SYCL                        )
declare -a OutputDirLocal=(      ${OutputDirRoot}/CUDA_Tiled_withhalocells   ${OutputDirRoot}/CUDA_Tiled_withouthalocells   ${OutputDirRoot}/OpenCL_without_halocells  ${OutputDirRoot}/OpenCL_with_halocells    ${OutputDirRoot}/SYCL_DPCCC_WITHOUT_HALO_CELLS  ${OutputDirRoot}/SYCL_DPCCC_WITH_HALO_CELLS    )
declare -a TargetMakeLocal=(     "cuda_tiling_with_halo_cells"               "cuda_tiling_without_halo_cells"               sciddicaTcl_without_halo_cells             sciddicaTcl_with_halo_cells               sciddicaTsycl_dpcpp_without_halo_cells          sciddicaTsycl_dpcpp_with_halo_cells            )
declare -a TargetRunMakeLocal=(  run_withhalocells                           run_withouthalocells                           run_without_halo_cells                     run_with_halo_cells                       run_dpcpp_without_halo_cells                    run_dpcpp_with_halo_cells                      )
declare -a NameTestLocal=(       run_straightforward_cuda_withhalocells      run_straightforward_cuda_withouthalocells      run_opencl_without_halo_cells              run_opencl_with_halo_cells                run_dpcpp_without_halo_cells_                   run_dpcpp_with_halo_cells_                     )


echo "${OutputDir[@]}"

for i in "${!OutputDir[@]}";
do
    mkdir -p ${OutputDir[$i]} 
done 

for i in "${!OutputDirLocal[@]}";
do
    echo "${OutputDirLocal[$i]} "
    mkdir -p ${OutputDirLocal[$i]} 
done 

for i in "${!FolderTest[@]}";
do
    make -C ${FolderTest[$i]}/ ${TargetMake[$i]} 
done 

for i in "${!FolderTestLocal[@]}";
do
    if [[ ${FolderTestLocal[$i]} != *"CUDA"* ]];
    then
        make -C ${FolderTestLocal[$i]}/ ${TargetMakeLocal[$i]} 
    fi
done 

echo "${block_size_x[@]}"
echo "${block_size_y[@]}"

#Memory Global Test
for t in  "${!FolderTest[@]}";#4 8 16 32
do
    for SIZE_X in  "${block_size_x[@]}";#4 8 16 32
    do
        for SIZE_Y in "${block_size_y[@]}";
        do
            for (( test=0; test<$numberoftests; test++))
            do
                cd ${FolderTest[$t]} && make ${TargetRunMake[$t]} BLOCK_SIZE_X=${SIZE_X} BLOCK_SIZE_Y=${SIZE_Y} | tee ${homeFolder[$t]}/${OutputDir[$t]}/log_${NameTest[$t]}${SIZE_X}_${SIZE_Y}_${test} && cd ${homeFolder[$t]} 
            done 
        done
    done
done

#Memory Local Test
for t in  "${!FolderTestLocal[@]}";
do
    for TILE_SIZE in "${tile_size[@]}"
    do
        for (( test=0; test<$numberoftests; test++))
        do
            if [[ ${FolderTestLocal[$t]} == *"CUDA"* ]];
            then
                echo "CUDA"
                make -C ${FolderTestLocal[$t]}/ ${TargetMakeLocal[$t]} TILE_SIZE=${TILE_SIZE} 
                echo " make -C ${FolderTestLocal[$t]}/ ${TargetMakeLocal[$t]} TILE_SIZE=${TILE_SIZE}  "
                cd ${FolderTestLocal[$t]} && make ${TargetRunMakeLocal[$t]} | tee ../${OutputDirLocal[$t]}/log_${NameTestLocal[$t]}_${TILE_SIZE}_${test} && cd ..
            else
                echo "SYCL"
                cd ${FolderTestLocal[$t]} && make ${TargetRunMakeLocal[$t]} TILE_SIZE=${TILE_SIZE} | tee ../${OutputDirLocal[$t]}/log_${NameTestLocal[$t]}_${TILE_SIZE}_${test} && cd ..
            fi
        done
    done
done




# Gather Output log results
declare -a outputResultsGlobal
declare -a block_sizes
declare -a outputCUDA
declare -a outputSYCL
declare -a outputSYCL_cols_major

for SIZE_X in  "${block_size_x[@]}";
do
    for SIZE_Y in "${block_size_y[@]}";
    do
        block_sizes+=(${SIZE_X}_${SIZE_Y})
    done
done
echo "${block_sizes[@]}"

count=0
strSYCL="SYCL"
for t in  "${!FolderTest[@]}";
do
    strtmp1=${FolderTest[$t]}
    if [[ $strtmp1 == *"$strSYCL"* ]];
    then
        for SIZE_X in  "${block_size_x[@]}";
        do
            for SIZE_Y in "${block_size_y[@]}";
            do
                declare -a timesResults
                for (( test=0; test<$numberoftests; test++))
                do
                    output=$(grep -hnra "time" ${OutputDir[$t]}//log_${NameTest[$t]}${SIZE_Y}_${SIZE_X}_${test})
                    stringarray=($output)
                    timesResults+=(${stringarray[2]})
                done
                min=${timesResults[0]}
                for i in "${timesResults[@]}"; do
                  min=$(echo "if($i<$min) $i else $min" | bc | awk '{printf "%f", $0}')
                done 
                outputResultsGlobal[${count}]+=${min}
                ((count=count+1)) 
                unset timesResults
            done
        done
    else
        for SIZE_X in  "${block_size_x[@]}";
        do
            for SIZE_Y in "${block_size_y[@]}";
            do
                declare -a timesResults
                for (( test=0; test<$numberoftests; test++))
                do
                    output=$(grep -hnra "time" ${OutputDir[$t]}//log_${NameTest[$t]}${SIZE_X}_${SIZE_Y}_${test})
                    stringarray=($output)
                    timesResults+=(${stringarray[2]})
                done
                min=${timesResults[0]}
                for i in "${timesResults[@]}"; do
                  min=$(echo "if($i<$min) $i else $min" | bc | awk '{printf "%f", $0}')
                done
                outputResultsGlobal[${count}]+=${min}
                ((count=count+1)) 
                unset timesResults
            done
        done
    fi
done

# echo "count: ${count}"
# for ((i=0; i<$count;i++))
# do
#     echo "${outputResultsGlobal[$i]}"
# done

len1=${#block_size_x[@]}
len2=${#block_size_y[@]}
dimOneTests=0
((dimOneTests=len1*len2))
echo "$dimOneTests"

echo -n "BLOCK_SIZE_(X_Y);" > ${OutputFileName}
for i in "${!NameCSVOUTPUT[@]}"; do
    echo -n "${NameCSVOUTPUT[$i]};" >> ${OutputFileName}
done
echo "" >> ${OutputFileName}

for i in "${!block_sizes[@]}"; do 
    echo -n "${block_sizes[$i]};" >> ${OutputFileName}
    for t in "${!NameTest[@]}"; 
    do
        ((index=t*dimOneTests+i))
        echo -n "${outputResultsGlobal[$index]};" >> ${OutputFileName}
    done
    echo "" >> ${OutputFileName}
done

echo "Results saved in ${OutputFileName}"





# Gather Output log results
declare -a outputResultsLocal
declare -a tile_sizes
declare -a outputCUDA
declare -a outputSYCL
declare -a outputSYCL_cols_major

for tile in "${tile_size[@]}";
do
    tile_sizes+=(${tile})
done
echo "${tile_sizes[@]}"

count=0
strSYCL="SYCL"
for t in  "${!FolderTestLocal[@]}";
do
    strtmp1=${FolderTestLocal[$t]}
    for tile in  "${tile_size[@]}";
    do
        declare -a timesResults
        for (( test=0; test<$numberoftests; test++))
        do
            output=$(grep -hnra "time" ${OutputDirLocal[$t]}//log_${NameTestLocal[$t]}_${tile}_${test})
            stringarray=($output)
            timesResults+=(${stringarray[2]})
        done
        min=${timesResults[0]}
        for i in "${timesResults[@]}"; do
          min=$(echo "if($i<$min) $i else $min" | bc | awk '{printf "%f", $0}')
        done 
        outputResultsLocal[${count}]+=${min}
        ((count=count+1)) 
        unset timesResults
    done
done

# echo "count: ${count}"
# for ((i=0; i<$count;i++))
# do
#     echo "${outputResultsGlobal[$i]}"
# done

len1=${#tile_size[@]}
dimOneTests=0
((dimOneTests=len1))
echo "$dimOneTests"

echo -n "TILE_SIZE;" > ${OutputFileNameLocal}
for i in "${!NameCSVOUTPUTLocal[@]}"; do
    echo -n "${NameCSVOUTPUTLocal[$i]};" >> ${OutputFileNameLocal}
done
echo "" >> ${OutputFileNameLocal}

for i in "${!tile_size[@]}"; do 
    echo -n "${tile_size[$i]};" >> ${OutputFileNameLocal}
    for t in "${!NameTestLocal[@]}"; 
    do
        ((index=t*dimOneTests+i))
        echo -n "${outputResultsLocal[$index]};" >> ${OutputFileNameLocal}
    done
    echo "" >> ${OutputFileNameLocal}
done

echo "Results saved in ${OutputFileNameLocal}"
