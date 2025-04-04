for patient in 20
do
    output_dir="/data/che/TRIM/HNSCC/output/holdout${patient}"

    # Create the directory if it doesn't exist
    mkdir -p "${output_dir}"
    echo "Starting training for patient ${patient}..."

    python -u trim.py \
        --heldout_patient=${patient} \
        --model_name="holdout${patient}" \
        --batch_size=4096 \
        > "${output_dir}/training.log"

    echo "Training completed for patient ${patient}."
done
