# Sourced once by a runner after prepare_task. Preserve its existing cleanup and exit status.
if (( DRY_RUN == 0 )); then
    export LATENTPPI_REQUESTED_RECYCLES=${RECYCLE_COUNT:-}
    LATENTPPI_ATTEMPT_ID=$("$PYTHON_BIN" "$SCRIPT_DIR/record_run_env.py" start "$AF3_OUTPUT_DIR" "$JOB_NAME" "$JSON_PATH")
    export LATENTPPI_ATTEMPT_ID
    latentppi_environment_exit() {
        local original_status=$?
        "$PYTHON_BIN" "$SCRIPT_DIR/record_run_env.py" finish "$AF3_OUTPUT_DIR" "$JOB_NAME" "$JSON_PATH" "$original_status" "${STAGE:-unknown}" || echo 'WARNING: could not publish run environment' >&2
        # The condition transfers the original status to cleanup's initial `$?`.
        if (exit "$original_status"); then cleanup; else cleanup; fi
    }
    trap latentppi_environment_exit EXIT
fi
