#!/bin/bash
#title          :chopin2 wrapper
#description    :Feature importance with random decomposition and feature selection
#author         :Fabio Cumbo (fabio.cumbo@gmail.com)
#==================================================================================

: '
How to use this script

Consider the following configuration:
    > DATASET="/path/to/dataset.csv"
    > FIELDSEP=","
    > OUTDIR="/path/to/out/"
    > CHUNKS=10
    > DIMENSIONALITY=10000
    > LEVELS=100
    > RETRAIN=10
    > FOLDS=5
    > GROUPMIN=1
    > ACCTHRES=40.0
    > ACCUNCERT=5.0
    > NPROC=8
    > XARGSNPROC=2
    > MAXITER=10

Run this script as below:
    > sh starter.sh $DATASET $FIELDSEP $OUTDIR $CHUNKS \
                    $DIMENSIONALITY $LEVELS $RETRAIN $FOLDS \
                    $GROUPMIN $ACCTHRES $ACCUNCERT \
                    $NPROC $XARGSNPROC $MAXITER
 '

DATASET=$1          # Path to the dataset file
FIELDSEP=$2         # Field separator
OUTDIR=$3           # Path to the output folder
CHUNKS=$4           # Split the input dataset into chunks

DIMENSIONALITY=$5   # HD dimensionality
LEVELS=$6           # HD levels
RETRAIN=$7          # Retraining iterations
FOLDS=$8            # K-folds cross-validation
GROUPMIN=$9         # Minimum number of features
ACCTHRES=${10}      # Threshold on the accuracy
ACCUNCERT=${11}     # Uncertainty threshold on the accuracy

NPROC=${12}         # Make the single run multiprocessing  
XARGSNPROC=${13}    # Run multiple instances of chopin2 in parallel

MAXITER=${14}       # Maximum number of tests

# Format seconds in human-readable format
# Credits: https://unix.stackexchange.com/a/27014
displaytime () {
    D=$(bc <<< "${1}/60/60/24") # Days
    H=$(bc <<< "${1}/60/60%24") # Hours
    M=$(bc <<< "${1}/60%60")    # Minutes
    S=$(bc <<< "${1}%60")       # Seconds
    if [[ "$D" -gt "0" ]]; then printf '%s days ' "$D"; fi
    if [[ "$H" -gt "0" ]]; then printf '%s hours ' "$H"; fi
    if [[ "$M" -gt "0" ]]; then printf '%s minutes ' "$M"; fi
    if [[ "$D" -gt "0" ]] || [[ "$H" -gt "0" ]] || [[ "$M" -gt "0" ]]; then printf 'and '; fi
    printf '%s seconds\n' "$S"
}

# Split a dataset and run the feature selection with chopin2 in parallel
run () {
    : '
    How to use this function

    Consider the following configuration:
        > RUN_DATASET="/path/to/dataset.csv"
        > RUN_FIELDSEP=","
        > RUN_OUTDIR="/path/to/out1/"
        > RUN_CHUNKS=10
        > RUN_DIMENSIONALITY=10000
        > RUN_LEVELS=100
        > RUN_RETRAIN=10
        > RUN_FOLDS=5
        > RUN_GROUPMIN=1
        > RUN_ACCTHRES=40.0
        > RUN_ACCUNCERT=5.0
        > RUN_NPROC=8
        > RUN_XARGSNPROC=2
        > RUN_SEED=0
        > RUN_SELECTIONS=/path/to/out0/

    First run chopin2 on your dataset as a single run, feature selection disabled:
        > chopin2 --dataset $RUN_DATASET --fieldsep $RUN_FIELDSEP --dimensionality $RUN_DIMENSIONALITY \
                  --levels $RUN_LEVELS --retrain $RUN_RETRAIN --stop --crossv_k $RUN_FOLDS \
                  --dump --cleanup --nproc $RUN_NPROC --verbose

    Then, run this script by randomly grouping features (groups have RUN_CHUNKS=10 features 
    at most in this example), feature selection enabled:
        > sh wrapper.sh $RUN_DATASET $RUN_FIELDSEP $RUN_OUTDIR $RUN_CHUNKS \
                        $RUN_DIMENSIONALITY $RUN_LEVELS $RUN_RETRAIN $RUN_FOLDS \
                        $RUN_GROUPMIN $RUN_ACCTHRES $RUN_ACCUNCERT \
                        $RUN_NPROC $RUN_XARGSNPROC $RUN_SEED \
                        $RUN_SELECTIONS

    Keep running this wrapper N times until the number of selected features does not change, 
    by reassigning RUN_OUTDIR and RUN_SELECTIONS at each run:
        > RUN_OUTDIR="/path/to/out2/"; RUN_SELECTIONS=/path/to/out1/; # sh wrapper.sh ...
        > RUN_OUTDIR="/path/to/out3/"; RUN_SELECTIONS=/path/to/out2/; # sh wrapper.sh ...
        ...
        > RUN_OUTDIR="/path/to/out(N)/"; RUN_SELECTIONS=/path/to/out(N-1))/; # sh wrapper.sh ...

    Finally, run this wrapper one more time by changing the RUN_CHUNKS value to the 
    total number of features selected with the last previous iteration (21 in this examples):
        > RUN_OUTDIR="/path/to/out(N+1)/"; RUN_SELECTIONS=/path/to/out(N))/; RUN_CHUNKS=21; # sh wrapper.sh ...
    '

    RUN_DATASET=$1          # Path to the dataset file
    RUN_FIELDSEP=$2         # Field separator
    RUN_OUTDIR=$3           # Path to the output folder
    RUN_CHUNKS=$4           # Split the input dataset into chunks

    RUN_DIMENSIONALITY=$5   # HD dimensionality
    RUN_LEVELS=$6           # HD levels
    RUN_RETRAIN=$7          # Retraining iterations
    RUN_FOLDS=$8            # K-folds cross-validation
    RUN_GROUPMIN=$9         # Minimum number of features
    RUN_ACCTHRES=${10}      # Threshold on the accuracy
    RUN_ACCUNCERT=${11}     # Uncertainty threshold on the accuracy

    RUN_NPROC=${12}         # Make the single run multiprocessing  
    RUN_XARGSNPROC=${13}    # Run multiple instances of chopin2 in parallel

    RUN_SEED=${14}          # Seed for randomly grouping features

    RUN_SELECTIONS=${15}    # Path to a previous run folder with chunk and chopin2 selections

    # Create the output folder if it does not exist
    mkdir -p $RUN_OUTDIR

    printf 'Decomposing dataset into chunks with %s features at most\n' "$RUN_CHUNKS"
    printf '\t%s\n\n' "$RUN_OUTDIR"

    # Assuming that the python script is located in the same folder of this wrapper
    python3 random_decomposition.py --dataset "$RUN_DATASET" \
                                    --fieldsep "$RUN_FIELDSEP" \
                                    --outdir "$RUN_OUTDIR" \
                                    --chunks "$RUN_CHUNKS" \
                                    --selections "$RUN_SELECTIONS"

    NCHUNKS=`find "${RUN_OUTDIR}" -type f -iname "dataset_*.txt" -follow | wc -l`
    printf 'Running chopin2 on %s chunks\n' "$NCHUNKS"
    CHOPIN2_START_TIME="$(date +%s.%3N)"

    find ${RUN_OUTDIR} \
        -type f -iname "dataset_*.txt" -follow | xargs -n 1 -P $RUN_XARGSNPROC -I {} bash -c \
        'INPUT={}; \
         printf "\t%s\n" "${INPUT}"; \
         if [ ! -f "$(dirname "${INPUT}")/selection.txt" ] ; then \
            LHD="$(dirname '"${RUN_DATASET}"')/levels_bufferHVs_'"${RUN_DIMENSIONALITY}"'_'"${RUN_LEVELS}"'.pkl"; \
            if [ -f "$LHD" ]; then \
                ln -s "$(realpath $LHD)" "$(dirname "${INPUT}")"; \
            fi ; \
            chopin2 --dataset "${INPUT}" \
                    --fieldsep '\""${RUN_FIELDSEP}\""' \
                    --dimensionality '"${RUN_DIMENSIONALITY}"' \
                    --levels '"${RUN_LEVELS}"' \
                    --retrain '"${RUN_RETRAIN}"' \
                    --stop \
                    --crossv_k '"${RUN_FOLDS}"' \
                    --select_features \
                    --group_min '"${RUN_GROUPMIN}"' \
                    --accuracy_threshold '"${RUN_ACCTHRES}"' \
                    --accuracy_uncertainty_perc '"${RUN_ACCUNCERT}"' \
                    --dump \
                    --cleanup \
                    --nproc '"${RUN_NPROC}"' \
                    --verbose >> "$(dirname "${INPUT}")/chopin2.log" 2>&1 ; \
         fi'

    CHOPIN2_END_TIME="$(date +%s.%3N)"
    CHOPIN2_ELAPSED="$(bc <<< "${CHOPIN2_END_TIME}-${CHOPIN2_START_TIME}")"
    printf '\nTotal elapsed time: %s\n\n' "$(displaytime ${CHOPIN2_ELAPSED})"

    printf 'Selected features:\n\n'

    find ${RUN_OUTDIR} \
        -type f -iname "selection.txt" -follow | xargs -n 1 -I {} bash -c \
        'INPUT={}; \
         printf "%s\t%s\n" "$INPUT" "$(tail -n +4 $INPUT | grep -v "^.$" | wc -l)";'
}

for SEED in $(seq 1 $MAXITER); do
    # Create the output folder for the current test
    SEEDPATH="$OUTDIR/seed$SEED"
    mkdir -p "$SEEDPATH"

    # Run counter
    RUNCOUNT=0

    # Number of selected features in the previous run
    PREV_FEATURES=0

    while true; do
        # Define the folder for the current run
        RUN_DIR="$SEEDPATH/run$RUNCOUNT"
        mkdir -p "$RUN_DIR"

        # Define the previous run directory
        PREV_RUN_DIR="$SEEDPATH/run$((RUNCOUNT-1))"

        # Assume wrapper.sh is located in the same forlder of this script
        sh wrapper.sh $DATASET $FIELDSEP $RUN_DIR $CHUNKS \
                      $DIMENSIONALITY $LEVELS $RETRAIN $FOLDS \
                      $GROUPMIN $ACCTHRES $ACCUNCERT \
                      $NPROC $XARGSNPROC $SEED \
                      $PREV_RUN_DIR
        
        # Count the number of selected features
        selections=($(find ${RUN_DIR} -type f -name "selection.txt"))
        # Number of selected features in the current run
        CURR_FEATURES=0
        for selection in "${selections[@]}"; do
            FEATURES=$(tail -n +4 $selection | grep -v "^.$" | wc -l)
            CURR_FEATURES=$((CURR_FEATURES+FEATURES))
        done

        # Break if the number of selected features is the same compared to the previous run
        if [[ "${PREV_FEATURES}" -eq "${CURR_FEATURES}" ]]; then
            break
        fi

        # Take track of the current number of selected features
        PREV_FEATURES=$CURR_FEATURES

        # Increment run counter
        RUNCOUNT=$((RUNCOUNT+1))
    done

    # Define the previous and last run folder path
    PREV_RUN_DIR="$SEEDPATH/run$RUNCOUNT"
    LAST_RUN_DIR="$SEEDPATH/run$((RUNCOUNT+1))"

    # Run the wrapper again considering all the selected features only
    sh wrapper.sh $DATASET $FIELDSEP $LAST_RUN_DIR $PREV_FEATURES \
                  $DIMENSIONALITY $LEVELS $RETRAIN $FOLDS \
                  $GROUPMIN $ACCTHRES $ACCUNCERT \
                  $NPROC $XARGSNPROC $SEED \
                  $PREV_RUN_DIR
done

# Finally report the number of occurrences of the features as selected features
python3 select_top_features.py --basepath "$OUTDIR"
