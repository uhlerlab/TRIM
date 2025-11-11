for patient in 20
do
    output_dir="your_output_directory/holdout${patient}"

    # Create the directory if it doesn't exist
    mkdir -p "${output_dir}"
    echo "Starting training for patient ${patient}..."

    python -u trim.py \
        --heldout_patient=${patient} \
        --model_name="holdout${patient}" \
        > "${output_dir}/training.log"

    echo "Training completed for patient ${patient}."
done
