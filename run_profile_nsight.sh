#!/bin/bash

if [ "$1" != "-gpu" ] || [ "$2" == "" ] || [ "$3" != "-block_size" ] || [ "$4" == "" ] || [ "$5" != "-type" ] || [ "$6" == "" ]  || [ "$7" != "-folder_name" ] || [ "$8" == "" ] || [ "$9" != "-cuda_executable" ] || [ "${10}" == "" ] || [ "${11}" != "-sycl_executable" ] || [ "${12}" == "" ]
then
  echo "$1 $2 $3 $4 $5 $6 $7 $8 $9 $10 $11 $12"
  echo "Usage: sh -gpu <gpu_id>"
  echo ""
  echo "Examples:"
  echo "  test on gpu 0 with block size 16x16:"
  echo "    bash $0 -gpu 0 -block_size 16 -type standard -folder_name standard -cuda_executable sciddicaTcudaStraightforward -sycl_executable sciddicaTsycl_dpcpp"
  echo ""
  echo "  test on gpu 1 with block_size 8x8:"
  echo "    bash $0 -gpu 1 -block_size 8 -type standard -folder_name standard -cuda_executable sciddicaTcudaStraightforward -sycl_executable sciddicaTsycl_dpcpp"
  echo ""
  exit 0
fi

if [ "$6" == "standard" ]
then
    parameterMetrics="../data/tessina_header.txt ../data/tessina_dem.txt ../data/tessina_source.txt ./tessina_output_cuda 100"
    parameterRun="../data/tessina_header.txt ../data/tessina_dem.txt ../data/tessina_source.txt ./tessina_output_cuda 4000"
fi

if [ "$6" == "stresstest" ]
then
    parameterMetrics="../data/tessina_headerX16.txt ../data/tessina_demX16.txt ../data/tessina_sourceX16.txt ./tessina_output_OpenCL 100"
    parameterRun="../data/tessina_headerX16.txt ../data/tessina_demX16.txt ../data/tessina_sourceX16.txt ./tessina_output_OpenCL 64000"
fi

#$6 type
#$8 folderName
#$10 cuda_executable
#$12 sycl_executable

HOSTNAME=`echo $HOSTNAME`
echo "HOSTNAME $HOSTNAME"
source ./setenv_$HOSTNAME.txt
export CUDA_VISIBLE_DEVICES=$2

OutputDir=./LogProfile_$8_gpu_$2_block_size_$4_hostname_${HOSTNAME}_`date +"%Y-%m-%d_%T"`
mkdir -p ${OutputDir}
OutputDirResults=$OutputDir/Results
mkdir -p ${OutputDirResults}

delimiter=";"

logfileprofilmetricsCUDA=./${OutputDir}/log_sciddicaTcuda_naive_oi.csv
logfileprofilruntimeCUDA=./${OutputDir}/log_sciddicaTcuda_naive_flop.csv
executableDirCUDA=./sciddicaT_nolibs_CUDA/
executableStrCUDA=" ./${10} $parameterMetrics $4 $4 && cd .."
executableStrRuntimeCUDA=" ./${10} $parameterRun $4 $4 && cd .."


logfileprofilmetricsSYCL=./${OutputDir}/log_sciddicaTsycl_naive_oi.csv
logfileprofilruntimeSYCL=./${OutputDir}/log_sciddicaTsycl_naive_flop.csv
executableDirSYCL=./sciddicaT_nolibs_SYCL/
executableStrSYCL=" ./${12} $parameterMetrics $4 $4 && cd .."
executableStrRuntimeSYCL=" ./${12} $parameterRun $4 $4 && cd .."


declare -a kernelsCUDA=( sciddicaTFlowsComputation sciddicaTWidthUpdate sciddicaTResetFlows )
declare -a kernelsSYCL=( Compute Update Reset )
declare -a metrics=( sm__sass_thread_inst_executed_op_dadd_pred_on.sum sm__sass_thread_inst_executed_op_dfma_pred_on.sum  sm__sass_thread_inst_executed_op_dmul_pred_on.sum sm__sass_thread_inst_executed_op_fadd_pred_on.sum sm__sass_thread_inst_executed_op_ffma_pred_on.sum  sm__sass_thread_inst_executed_op_fmul_pred_on.sum sm__sass_thread_inst_executed_op_hadd_pred_on.sum sm__sass_thread_inst_executed_op_hfma_pred_on.sum  sm__sass_thread_inst_executed_op_hmul_pred_on.sum sm__inst_executed_pipe_tensor.sum dram__bytes.sum lts__t_bytes.sum l1tex__t_bytes.sum)

echo "Profiling sciddicaTcuda_naive"
echo "..."

########## CUDA #############

 
profileStrCUDA="cd ${executableDirCUDA} && nv-nsight-cu-cli  --summary per-kernel --csv  --log-file ../${logfileprofilmetricsCUDA} --metrics "
for m in ${metrics[@]}
do
    profileStrCUDA+="$m,"
done
profileStrCUDA+=$executableStrCUDA
echo "Running $profileStrCUDA"
eval ${profileStrCUDA}
echo ""


profileStrRuntimeCUDA="cd ${executableDirCUDA} && nv-nsight-cu-cli --csv --summary per-kernel --log-file ../${logfileprofilruntimeCUDA} --metric  sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second ${executableStrRuntimeCUDA}"
echo "Running $profileStrRuntimeCUDA"
eval $profileStrRuntimeCUDA
echo ""

########## END  CUDA    #############

########## SYCL         #############

profileStrSYCL="cd ${executableDirSYCL} && nv-nsight-cu-cli  --summary per-kernel --csv  --log-file ../${logfileprofilmetricsSYCL} --metrics "
for m in ${metrics[@]}
do
    profileStrSYCL+="$m,"
done
profileStrSYCL+=$executableStrSYCL
echo "Running $profileStrSYCL"
eval $profileStrSYCL
echo ""

profileStrRuntimeSYCL="cd ${executableDirSYCL} && nv-nsight-cu-cli --csv --summary per-kernel --log-file ../${logfileprofilruntimeSYCL} --metric  sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second ${executableStrRuntimeSYCL}"
echo "Running $profileStrRuntimeSYCL"
eval $profileStrRuntimeSYCL
echo " "

########## END SYCL      #############

echo "Done"
echo "Results saved in ${OutputDir}/Results"
#echo "Results SYCL saved in ${logfileprofilruntimeSYCL}"

###############  CUDA NVPROF ##################
for i in "${kernelsCUDA[@]}"
do
    resultsFileNameKernel="${OutputDirResults}/results_cuda_${i}.csv"
    echo "${i}${delimiter}Min${delimiter}Max${delimiter}Avg" > $resultsFileNameKernel
    outputKernel=$(grep -hnr "${i}" ${logfileprofilmetricsCUDA})
    tmpFileName="${OutputDirResults}/tmpFile.txt"
    echo "" > "${tmpFileName}"
    for t in "${outputKernel[@]}"; 
    do
        echo "${t} \n">> "${tmpFileName}"
    done 

    #metricsUnique=($(printf "%s\n" "${metrics[@]}" | sort -u | tr '\n' ' '))
     
    for m in "${metrics[@]}"
    do
        outputMetric=$(grep -hnr "${m}" ${tmpFileName})
        len=${#outputMetric[@]}
        #delete endline
        for o in "${!outputMetric[@]}"
        do
            outputMetric[o]=$( echo "${outputMetric[o]}" | tr -d '\\n' )
        done
        r=${outputMetric[@]}
        dd=(${r//,/ })
        echo "${m}$delimiter${dd[-3]}$delimiter${dd[-2]}$delimiter${dd[-1]}$delimiter" >> $resultsFileNameKernel
    done
    rm $tmpFileName
done

for i in "${kernelsCUDA[@]}"
do 
    resultsFileNameKernel="${OutputDirResults}/results_cuda_${i}.csv"
#    unitsProfile=$(grep -hnr -E "${i}.*gpu__time_active.avg" ${logfileprofilruntimeCUDA})
#    r=${unitsProfile[@]}
#    dd=(${r//,/ })
#    echo "${i}${delimiter}gpu__time_active.avg${delimiter}(${dd[-4]})${delimiter}Avg(${dd[-1]})${delimiter}Min(${dd[-3]})${delimiter}Max(${dd[-2]})${delimiter}" >> $resultsFileNameKernel
    unitsProfile1=$(grep -hnr -E "${i}.*sm__cycles_elapsed.avg" ${logfileprofilruntimeCUDA})
    r1=${unitsProfile1[@]}
    dd1=(${r1//,/ })
    echo "${i}${delimiter}${delimiter}${delimiter}Avg${delimiter}Min${delimiter}Max${delimiter}" >> $resultsFileNameKernel
    echo "${i}${delimiter}sm__cycles_elapsed.avg${delimiter}time(${dd1[-4]})${delimiter}${dd1[-3]}${delimiter}${dd1[-2]}${delimiter}${dd1[-1]}${delimiter}" >> $resultsFileNameKernel
    unitsProfile2=$(grep -hnr -E "${i}.*sm__cycles_elapsed.avg.per_second" ${logfileprofilruntimeCUDA})
    r2=${unitsProfile2[@]}
    dd2=(${r//,/ })
    echo "${i}${delimiter}sm__cycles_elapsed.avg.per_second${delimiter}time(${dd2[-4]})${delimiter}${dd2[-3]}${delimiter}${dd2[-2]}${delimiter}${dd2[-1]}${delimiter}" >> $resultsFileNameKernel 
done
############### END CUDA NVPROF ##################


###############  SYCL DPCPP NVPROF ##################

for i in "${kernelsSYCL[@]}"
do 
    resultsFileNameKernel="${OutputDirResults}/results_sycl_${i}.csv"
    echo "${i}${delimiter}Min${delimiter}Max${delimiter}Avg" > $resultsFileNameKernel
    outputKernel=$(grep -hnr "${i}" ${logfileprofilmetricsSYCL})
    tmpFileName="${OutputDirResults}/tmpFile.txt"
    echo "" > "${tmpFileName}"
    for t in "${outputKernel[@]}"; 
    do
        echo "${t} \n">> "${tmpFileName}"
    done 
    for m in "${metrics[@]}"
    do
        outputMetric=$(grep -hnr "${m}" ${tmpFileName})
        len=${#outputMetric[@]}
        #delete endline
        for o in "${!outputMetric[@]}"
        do
            outputMetric[o]=$( echo "${outputMetric[o]}" | tr -d '\\n' )
        done
        r=${outputMetric[@]}
        dd=(${r//,/ })
        echo "${m}$delimiter${dd[-3]}$delimiter${dd[-2]}$delimiter${dd[-1]}$delimiter" >> $resultsFileNameKernel
    done
    rm $tmpFileName
done

for i in "${kernelsSYCL[@]}"
do 
    resultsFileNameKernel="${OutputDirResults}/results_sycl_${i}.csv"
#   unitsProfile=$(grep -hnr -hnr -E "${i}.*gpu__time_active.avg" ${logfileprofilruntimeSYCL})
#    r=${unitsProfile[@]}
#    dd=(${r//,/ })
#    echo "${i}${delimiter}gpu__time_active.avg${delimiter}time(${dd[-4]})${delimiter}Avg(${dd[-3]})${delimiter}Min(${dd[-2]})${delimiter}Max(${dd[-1]})${delimiter}" >> $resultsFileNameKernel 
    unitsProfile1=$(grep -hnr -hnr -E "${i}.*sm__cycles_elapsed.avg" ${logfileprofilruntimeSYCL})
    r1=${unitsProfile1[@]}
    dd1=(${r1//,/ })
    echo "${i}${delimiter}${delimiter}${delimiter}Avg${delimiter}Min${delimiter}Max${delimiter}" >> $resultsFileNameKernel
    echo "${i}${delimiter}sm__cycles_elapsed.avg${delimiter}time(${dd1[-4]})${delimiter}${dd1[-3]}${delimiter}${dd1[-2]}${delimiter}${dd1[-1]}${delimiter}" >> $resultsFileNameKernel 
    unitsProfile2=$(grep -hnr -hnr -E "${i}.*sm__cycles_elapsed.avg.per_second" ${logfileprofilruntimeSYCL})
    r2=${unitsProfile2[@]}
    dd2=(${r2//,/ })
    echo "${i}${delimiter}sm__cycles_elapsed.avg.per_second${delimiter}time(${dd2[-4]})${delimiter}${dd2[-3]}${delimiter}${dd2[-2]}${delimiter}${dd2[-1]}${delimiter}" >> $resultsFileNameKernel 
done

###############  END SYCL DPCPP NVPROF ##################
